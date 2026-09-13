"""Unit checks for the continuous bank curve (owner spec 2026-08-15: 160 sims on a
full bank, square-root decline -- slow early, steep late -- to raw policy at the 40 s
floor). Loads the CANONICAL template turn_search.py the way test_root_veto does."""
import importlib.util
import sys

import os
TEMPLATE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "agent", "turn_search.py"))
spec = importlib.util.spec_from_file_location("turn_search_under_test", TEMPLATE)
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)

failures = []


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        failures.append(name)


def budget_at(bank):
    return ts.search_budget({"remainingOverageTime": bank})


# -- anchors ---------------------------------------------------------------------------
check("full 600 s bank -> SIMS_TOP (256)", budget_at(600.0)[0] == 256)
check("top per-move cap is exactly 9.0 s", abs(budget_at(600.0)[1] - 9.0) < 1e-9)
sims_400 = budget_at(400.0)[0]
check("400 s bank -> ~208 (held high early)", sims_400 in (200, 208))
sims_300 = budget_at(300.0)[0]
check("half bank (300 s) -> ~176 (still deep)", sims_300 in (168, 176))
sims_100 = budget_at(100.0)[0]
check("100 s bank -> ~80 (steep late decline)", sims_100 in (80, 88))

# -- floor and tail --------------------------------------------------------------------
check("40 s bank -> raw (None)", budget_at(40.0) is None)
check("below floor -> raw (None)", budget_at(12.0) is None)
check("sub-16-sim tail (~41 s) -> raw (None)", budget_at(41.0) is None)
low = budget_at(50.0)
check("50 s bank still searches (>=16 sims)", low is not None and low[0] >= 16)

# -- shape properties ------------------------------------------------------------------
banks = [600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 75, 60, 50]
sims = [budget_at(float(b))[0] for b in banks]
check("monotone non-increasing as the bank drains",
      all(a >= b for a, b in zip(sims, sims[1:])))
check("all budgets quantized to multiples of 8", all(s % 8 == 0 for s in sims))
check("no cliff bigger than 40 sims between samples (old ladder halved by 128)",
      all(a - b <= 40 for a, b in zip(sims, sims[1:])))
early_drop = sims[0] - sims[4]            # 600 -> 400: first 200 s of drain
late_drop = budget_at(250.0)[0] - budget_at(50.0)[0]   # 250 -> 50: last 200 s
check("decays slower early than late (owner shape)", early_drop < late_drop)

# -- deadline scaling ------------------------------------------------------------------
check("deadline = base + per-sim * sims at every sample",
      all(abs(budget_at(float(b))[1] - (0.4 + 0.03359375 * budget_at(float(b))[0])) < 1e-9
          for b in banks))
check("top deadline matches the old 9 s profile", budget_at(600.0)[1] <= 9.0)

# -- input handling --------------------------------------------------------------------
check("bank above 600 clamps to the top", budget_at(700.0) == budget_at(600.0))
ts._clock["episode_start"] = None
import time as _time
ts._clock["episode_start"] = _time.time()
fallback = ts.search_budget(None)          # wall-clock fallback: fresh episode
check("wall-clock fallback works and starts near the top",
      fallback is not None and fallback[0] >= 240)
check("non-numeric remainingOverageTime falls back",
      ts.search_budget({"remainingOverageTime": "bad"}) is not None)

print()
print("FAILURES:", failures if failures else "none -- all pass")
sys.exit(1 if failures else 0)
