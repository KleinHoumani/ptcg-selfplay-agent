"""Unit matrix for the search ROOT VETO (owner go 2026-08-15): a root child whose Q
converged to a certain loss cannot win on visit count (the episode-93157396 override)."""
import importlib.util
import sys

import os
BUNDLE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "agent"))
TEMPLATE = os.path.join(BUNDLE, "turn_search.py")
sys.path.insert(0, BUNDLE)                     # bundle-local src for imports

spec = importlib.util.spec_from_file_location("ts_veto", TEMPLATE)
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)

failures = []


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        failures.append(name)


def root_with(visits, totals, priors, allowed=None):
    node = ts._Node(0, "decision", 0.5, priors=list(priors),
                    n_children=len(visits), allowed=allowed)
    node.visits = list(visits)
    node.total = list(totals)
    return node


# The ladder incident shape: high-prior child, most visits, Q converged to -1.
root = root_with(visits=[120, 80, 40], totals=[-120.0, 40.0, 22.0],
                 priors=[0.69, 0.2, 0.1])
check("veto: converged -1 child loses despite most visits",
      ts._pick_root_child(root) == 1)

# Q below the bar but visit count under the floor: too early to condemn.
root = root_with(visits=[5, 3], totals=[-5.0, 1.2], priors=[0.7, 0.3])
check("veto: under the visit floor nothing is vetoed",
      ts._pick_root_child(root) == 0)

# A merely-bad child (Q=-0.5) is not vetoed -- only converged certain losses.
root = root_with(visits=[100, 60], totals=[-50.0, 30.0], priors=[0.6, 0.4])
check("veto: Q=-0.5 is not a veto", ts._pick_root_child(root) == 0)

# Genuinely lost position: every child ~-1 -> fall back to the normal pick.
root = root_with(visits=[90, 70], totals=[-88.0, -69.0], priors=[0.5, 0.5])
check("veto: all-vetoed falls back to most-visited",
      ts._pick_root_child(root) == 0)

# Unvisited children never divide by zero and never get vetoed.
root = root_with(visits=[50, 0], totals=[-50.0, 0.0], priors=[0.9, 0.1])
check("veto: unvisited child survives the veto and wins by fallback",
      ts._pick_root_child(root) == 1)

# The mask's allowed set still gates candidacy before the veto.
root = root_with(visits=[120, 80, 40], totals=[-120.0, 40.0, 22.0],
                 priors=[0.69, 0.2, 0.1], allowed={0, 2})
check("veto: allowed-set filtering composes (child 1 masked away)",
      ts._pick_root_child(root) == 2)

print()
print("FAILURES:", failures if failures else "none -- all pass")
sys.exit(1 if failures else 0)
