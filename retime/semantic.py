"""Traduction JA -> langue du .ass (NLLB) et similarité multilingue (LaBSE), sur GPU."""
from __future__ import annotations

import gc
import json
import threading

import numpy as np

from .paths import CACHE_DIR

TARGET_LANGS = {
    "Français": "fra_Latn",
    "English": "eng_Latn",
    "Español": "spa_Latn",
    "Deutsch": "deu_Latn",
    "Italiano": "ita_Latn",
    "Português": "por_Latn",
}
TRANSLATION_MODELS = {
    "NLLB-200 600M (rapide)": "facebook/nllb-200-distilled-600M",
    "NLLB-200 1.3B (meilleur)": "facebook/nllb-200-distilled-1.3B",
}
EMBED_MODEL = "sentence-transformers/LaBSE"

_lock = threading.Lock()
_CACHE_FILE = CACHE_DIR / "translations.json"


def _load_cache() -> dict:
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict) -> None:
    tmp = _CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_CACHE_FILE)


_models: dict = {}


def _local_first(loader, *args, **kw):
    """Charge depuis le cache local sans réseau ; télécharge seulement si absent."""
    try:
        return loader(*args, local_files_only=True, **kw)
    except Exception:  # noqa: BLE001
        return loader(*args, **kw)


def _get_translator(model_id: str):
    if _models.get("tr_id") != model_id:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _models.pop("tr", None)
        tok = _local_first(AutoTokenizer.from_pretrained, model_id, src_lang="jpn_Jpan")
        model = _local_first(AutoModelForSeq2SeqLM.from_pretrained, model_id, dtype=torch.float16).to("cuda").eval()
        _models["tr"], _models["tr_id"] = (tok, model), model_id
    return _models["tr"]


def get_embedder() -> "Embedder":
    if "emb" not in _models:
        _models["emb"] = Embedder()
    return _models["emb"]


def release() -> None:
    _models.clear()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def translate(texts: list[str], model_key: str, tgt: str = "fra_Latn", batch_size: int = 48,
              progress=None, cancel=None) -> list[str]:
    model_id = TRANSLATION_MODELS.get(model_key, model_key)
    with _lock:
        cache = _load_cache()
    keys = [f"{model_id}|{tgt}|{t}" for t in texts]
    todo = sorted({t for t, k in zip(texts, keys) if k not in cache and t.strip()}, key=len)
    if todo:
        import torch

        tok, model = _get_translator(model_id)
        bos = tok.convert_tokens_to_ids(tgt)
        with torch.inference_mode():
            for a in range(0, len(todo), batch_size):
                if cancel and cancel():
                    raise InterruptedError
                batch = todo[a:a + batch_size]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=160).to("cuda")
                gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=3,
                                     max_new_tokens=160, repetition_penalty=1.1)
                for src, out in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
                    cache[f"{model_id}|{tgt}|{src}"] = out.strip()
                if progress:
                    progress(min(1.0, (a + len(batch)) / len(todo)))
        with _lock:
            merged = _load_cache()
            merged.update(cache)
            _save_cache(merged)
    return [cache.get(k, "") for k in keys]


class Embedder:
    def __init__(self, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer

        self.model = _local_first(SentenceTransformer, EMBED_MODEL, device=device)
        self.model.half()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        uniq = list(dict.fromkeys(texts))
        emb = self.model.encode(uniq, batch_size=256, normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False).astype(np.float32)
        pos = {t: i for i, t in enumerate(uniq)}
        return emb[[pos[t] for t in texts]]
