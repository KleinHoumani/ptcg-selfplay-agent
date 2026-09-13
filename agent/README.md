# agent/

The four Python files of the deployed Kaggle bundle, copied from the Sylveon submission.
The Dragapult ex bundle shipped the same `main.py`, `router.py` and `search_client.py`;
its `turn_search.py` differs only in three budget constants (see below).

| File | Role |
|---|---|
| `main.py` | Entry point Kaggle executes. Loads `ppo.pt` and the specialists, encodes each observation with `src/game/state_encoder.py`, runs the search, applies the action rules, and returns the chosen option indices. Any failure inside a decision falls back to a legal random move. |
| `turn_search.py` | Determinized PUCT search on the game engine to the end of our turn: hidden cards sampled, coins as 50/50 chance nodes, shuffle-draw Supporters as sampled chance nodes, budget from the time bank. |
| `router.py` | Naive Bayes posterior over every decklist in the archetype pools, updated from the opponent's revealed cards; swaps a specialist in at 0.8 and out below 0.7. |
| `search_client.py` | A batched leaf evaluator for the search. Measured slower than the default path and never enabled; kept because `main.py` imports it. |

## Configuration lives in the checkpoint envelope

`ppo.pt` carries a metadata envelope next to the fp16 weights: the encoding version, the
model dims, the list of enabled action rules, and the search options (reveal sampling,
fetch branching). `main.py` reads that list at load, so the same code runs the Dragapult
bundle with 10 rules and the Sylveon bundle with 4.

## Per-deck search budget

`turn_search.py` computes `simulations = SIMS_TOP * ((bank - 40) / 560) ** SIMS_CURVE_EXPONENT`
and a per-move deadline of `simulations * DEADLINE_PER_SIM`. This copy holds the Sylveon
values; the Dragapult ex bundle changed three constants:

| Constant | Sylveon | Dragapult ex |
|---|---|---|
| `SIMS_TOP` | 256 | 192 |
| `SIMS_CURVE_EXPONENT` | 0.5 | 0.33 |
| `DEADLINE_PER_SIM` | 0.03359375 (9.0 s at the top) | 0.034375 (7.0 s at the top) |

## Files the bundle needs at runtime

`deck.csv`, `ppo.pt`, `models/<archetype>.pt`, `router_data.json`, `src/`, `cg/` (the
engine SDK) and `data/cards/`. The top-level README says where each of these comes from.
