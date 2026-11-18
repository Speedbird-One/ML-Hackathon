"""PathSnap: ML-based map matching (candidate scoring + HMM/Viterbi)."""
import os
import site

# Windows Python 3.14 compatibility: ensure C-extension directories are in DLL search path
if os.name == "nt":
    for sp in site.getsitepackages():
        for root, dirs, files in os.walk(sp):
            if any(f.endswith(".pyd") for f in files):
                try:
                    os.add_dll_directory(root)
                except Exception:
                    pass
