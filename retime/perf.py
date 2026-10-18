"""Réglages de performance Windows : pas de bridage EcoQoS (E-cores) et priorité relevée."""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]


def boost_process() -> None:
    """Sur les CPU hybrides (P-cores/E-cores), Windows 11 relègue les processus jugés en arrière-plan
    sur les E-cores à fréquence réduite : on s'en exclut explicitement pendant les traitements."""
    if os.name != "nt":
        return
    try:
        k32 = ctypes.windll.kernel32
        proc = k32.GetCurrentProcess()
        process_power_throttling = 4
        execution_speed, ignore_timer = 0x1, 0x4
        state = _PowerThrottlingState(1, execution_speed | ignore_timer, 0)
        k32.SetProcessInformation(proc, process_power_throttling, ctypes.byref(state), ctypes.sizeof(state))
        k32.SetPriorityClass(proc, 0x00008000)          # ABOVE_NORMAL_PRIORITY_CLASS
    except Exception:  # noqa: BLE001
        pass
