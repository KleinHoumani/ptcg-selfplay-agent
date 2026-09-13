"""The evaluation panel the trainer's probe plays against: the four official sample agents
plus a random agent.

To run probes (the periodic in-training probe, or `train_ppo.py --probe-only`), place the
four official sample agent files under data/official_samples/agents/ (dragapult_ex.py,
iono_s.py, mega_abomasnow_ex.py, mega_lucario_ex.py; each defines `agent` and `my_deck`).
Training runs without them when `--probe-every 0` is passed.
"""

import importlib.util
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = ROOT / "data" / "official_samples" / "agents"
OPPONENTS = ("dragapult_ex", "iono_s", "mega_abomasnow_ex", "mega_lucario_ex", "random")


def random_legal(observation):
    """A uniformly random legal answer to the pending select."""
    select = observation["select"]
    count = len(select["option"])
    take = min(select["maxCount"], count)
    if take < select["minCount"]:
        take = select["minCount"]
    return sorted(random.sample(range(count), take)) if count else []


def load_opponent(name):
    """-> (agent callable, deck list or None for the random agent)."""
    if name == "random":
        return (lambda observation: random_legal(observation)), None
    path = AGENTS_DIR / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"panel agent {name} not found at {path} (see eval_panel.py)")
    spec = importlib.util.spec_from_file_location(f"panel_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent, list(module.my_deck)
