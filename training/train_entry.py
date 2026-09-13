"""Entry point that pins the multiprocessing start method, then hands off to train_ppo.

WHY THIS FILE EXISTS instead of a one-line edit inside train_ppo.py: Windows defaults to
"spawn", Linux defaults to "fork". train_ppo builds its worker Pool AFTER the model is on
the GPU and the inference server is running, so on Linux a fork would hand all 16 workers
a copy-on-write view of the ~12 GB trainer plus a live CUDA context. On a box whose RAM
failure mode is a SILENT allocation death (2026-08-07), that is not something you want to
diagnose over SSH. Setting it out here keeps train_ppo.py byte-identical for every other
consumer of that file.

Cross-platform on purpose: "spawn" is already the Windows default, so this is a no-op
there and the file stays interchangeable with `python train_ppo.py`.

    python experiments/vast_port/train_entry.py <every train_ppo flag, unchanged>

The sys.path insert is at MODULE level, not inside __main__, because a spawned child
re-executes this module as __mp_main__ and must be able to import train_ppo to unpickle
the pool initializer.
"""
import multiprocessing
import sys
from pathlib import Path

TRAINER_DIR = Path(__file__).resolve().parent
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

if __name__ == "__main__":
    # force=True so this can never raise if some import already fixed the context.
    multiprocessing.set_start_method("spawn", force=True)
    import train_ppo

    train_ppo.main()
