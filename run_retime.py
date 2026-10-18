"""Point d'entrée de Retime (utilisé aussi par PyInstaller)."""
import multiprocessing
import os
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if getattr(sys, "frozen", False) and sys.stdout is None:   # exe sans console
        sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")
    from retime.perf import boost_process

    boost_process()
    if "--cli" in sys.argv:
        from retime.cli import main as cli_main

        sys.exit(cli_main(sys.argv[sys.argv.index("--cli") + 1:]))
    from retime.gui.app import main

    main()
