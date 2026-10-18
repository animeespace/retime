# Retime

Recale automatiquement un `.ass` sur la voix japonaise d'une vidéo/audio, en se servant de Whisper, de l'analyse
de la courbe sonore et de la comparaison de sens entre le japonais entendu et le texte du `.ass`.

**Lancer** : `dist\Retime\Retime.exe` (raccourci « Retime » sur le Bureau).

## Utilisation

1. Glisser la vidéo (`.mkv`, `.mp4`…) ou l'audio et le `.ass` dans la fenêtre. La piste japonaise est choisie
   automatiquement (modifiable dans « Piste audio »).
2. **Analyser et recaler** (F5).
3. Vérifier : la timeline montre le spectrogramme, la forme d'onde, la courbe d'énergie de la voix isolée (zones
   vertes = voix) et trois pistes : *Original* (orange), *Whisper JP* (bleu), *Nouveau* (vert).
   - Cliquer une ligne (tableau ou timeline) zoome dessus ; **Espace** écoute le nouveau timing,
     **Maj+Espace** l'original, **Échap** coupe, **N** saute à la prochaine ligne à vérifier.
   - La région verte se déplace/étire à la souris ; les temps sont aussi éditables dans le tableau.
   - Décocher une ligne garde son timing d'origine.
4. **Exporter le .ass recalé** (Ctrl+S). Par défaut : `<nom> retime.ass` à côté de l'original, qui n'est jamais
   écrasé sans confirmation. Seules les colonnes Start/End changent (styles, tags, encodage intacts).

**Traitement par lot** : ajouter les vidéos et les `.ass`. Ils sont appariés par numéro d'épisode.

## Réglages utiles

| Réglage | Effet |
|---|---|
| Lead-in / Lead-out | Avance du début / prolongation de la fin par rapport à la voix (30 / 80 ms par défaut, calibrés sur ton retime de MFB 11) |
| Garder si écart < | Ne touche pas les lignes déjà à moins de X ms (0 = tout recaler) |
| Lier si écart < | Prolonge une ligne jusqu'à la suivante si le trou est plus court |
| Caler sur les frames | Temps posés sur les frames de la vidéo (convention Aegisub) |
| Keyframes | Accroche aux images clés du flux (désactivé : sur tes raws MFB elles tombent toutes les 2 s, pas aux changements de plan) |
| Styles | *Recaler* = dialogue ; *Décaler seulement* = panneaux/typeset (lignes `\pos`/`\move` toujours décalées, jamais accrochées à la voix) |

## Fonctionnement

1. ffmpeg extrait la piste ; **Demucs** isole la voix et **Whisper large-v3** transcrit le japonais (en parallèle, GPU).
2. Courbe d'énergie de la voix isolée (pas de 10 ms) → attaques, retombées, creux.
3. Décalage global par corrélation voix ↔ sous-titres, détection des dérives de framerate (25 / 23,976…).
4. Phrases japonaises ↔ lignes du `.ass` par similarité multilingue (**LaBSE**) → suivi du décalage ligne par
   ligne (gère coupures et scènes retirées).
5. Chaque ligne est accrochée à l'attaque / la retombée la plus probable, puis post-traitement (lead-in/out,
   chevauchements, durée minimale, frames).
6. Le japonais entendu sur chaque ligne est traduit (**NLLB**) et comparé au texte du `.ass` → colonne « Sens »
   et confiance ; les lignes douteuses sont surlignées.

Les modèles (~10 Go) et le cache d'analyse (voix isolée + transcription, ~50 Mo/épisode) sont dans
`%LOCALAPPDATA%\Retime` (menu « ⋯ » pour l'ouvrir ou vider le cache). Relancer un épisode déjà analysé prend ~5-20 s.

## Développement

```powershell
.venv\Scripts\python.exe run_retime.py                       # lancer depuis les sources
.\build.ps1                                                  # reconstruire l'exe (-Clean pour un build complet)
dist\Retime\Retime.exe --cli VIDEO ASS SORTIE.ass --log log.txt   # mode sans interface
.venv\Scripts\python.exe tools\eval_mod.py VIDEO ASS ASS_RETIME_MANUEL   # précision vs un retime manuel
.venv\Scripts\python.exe tools\robust.py VIDEO ASS ASS_RETIME_MANUEL DOSSIER_TMP   # décalages artificiels
```

Environnement : Python 3.12, torch 2.11 cu128 (RTX série 50 / sm_120), faster-whisper 1.2 (CTranslate2 4.8),
PySide6 + pyqtgraph, PyInstaller.
