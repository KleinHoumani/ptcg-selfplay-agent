"""[ISOLATED EXPERIMENT -- safe to delete this whole folder]

PPO self-play FROM SCRATCH (2026-07-16): the competitor-validated regime. No search, no
BC, no panel data in training -- the GameStateTransformer trunk (fresh weights) plays
itself at ~70 games/s across the worker pool and learns policy + value by clipped PPO
with GAE. Terminal win/loss is the ONLY reward (prizes are features, never the objective).

Design (agreed 2026-07-16):
  - Synchronous iterations: the pool plays a block of games with frozen weights (pure
    on-policy), the GPU learner updates, weights ship back to the workers by version.
  - Both players share the net and BOTH players' decisions are trained (2x data/game).
  - Decks: corpus-sampled both sides (the metagame, not the panel bots' five decks).
  - Anti-cycling: a fraction of games seat a frozen past snapshot on one side; only the
    current side's decisions are trained. Current-vs-past win rate is logged.
  - Encoding matches FullEncoderGuide inference EXACTLY (full + rich + select-context +
    solver features, deck_counts for the mover) and checkpoints carry the guide's keys,
    so the trained trunk (or just its value head) drops into the search composition.
  - Health metrics per iteration: explained variance vs lambda-returns (the competitor's
    "EV should quickly reach 0.8+" gate) AND vs raw MC outcomes, entropy, approx KL,
    clip fraction, first-player win rate, current-vs-past win rate, games/s.

  smoke:  CG_DLL=... python experiments/selfplay_ppo/train_ppo.py --iterations 2 \
              --games-per-iter 24 --workers 8 --probe-every 0
  train:  CG_DLL=... python experiments/selfplay_ppo/train_ppo.py --run-name run1
  ei3 raw-policy baseline on the SAME probe:
          CG_DLL=... python experiments/selfplay_ppo/train_ppo.py --probe-only \
              --probe-ckpt experiments/ladder_bc/ladder_bc_v3_ei3.pt --probe-games 200
"""

import argparse
import bisect
import copy
import json
import multiprocessing
import os
import queue
import random
import sys
import threading
import time
from collections import Counter
from itertools import chain
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "plan_agent"))
sys.path.insert(0, str(HERE))          # search_gen / true_state, whoever imports us
os.environ.setdefault("CG_DLL", str(ROOT / "engine_src" / "build" / "cg.dll"))

from src.game.encode import OPTION_FEATURE_DIM, encode_option                      # noqa: E402
from src.game.encode_details import OPTION_EXTRA_DIM, encode_option_v3                  # noqa: E402
from src.tiebreak import argmax_tiebreak                                           # noqa: E402
from src.game.encode_full import (GLOBAL_FEATURE_DIM_FULL, NUM_ZONES_FULL,          # noqa: E402
                                  SELECT_CONTEXT_DIM, SOLVER_GLOBAL_DIM,
                                  TOKEN_FEATURE_DIM_FULL_RICH, encode_observation_full)
from src.game.encode_rich import (RICH_CARD_DIM, RICH_GLOBAL_DIM,                   # noqa: E402
                                  player_block, rich_state)
from src.game.encode_history import (CARD_VOCAB as V2_CARD_VOCAB,                        # noqa: E402
                                GLOBAL_FEATURE_DIM_V2, GLOBAL_FEATURE_DIM_V3,
                                NUM_ZONES_V2, NUM_ZONES_V3,
                                TOKEN_FEATURE_DIM_V2, TOKEN_FEATURE_DIM_V3,
                                encode_observation_v2, encode_observation_v3)
from src.game.encode_history_cached import CachedV2Encoder, CachedV3Encoder               # noqa: E402
from src.game import encode_selection, encode_inflight, state_encoder                                # noqa: E402
from src.game.engine_projection import engine_projected_hp                          # noqa: E402
from src.decks.card_knowledge import CardKnowledge                                  # noqa: E402
from src.game.action_history import ActionHistory                                   # noqa: E402
from src.models.transformer import (GameStateTransformer,                           # noqa: E402
                                    GameStateTransformerConfig)
import aux_head_labels                                                                  # noqa: E402

# --- Encoding switch. "v1" is the run6 encoding and stays the DEFAULT so existing runs and
# checkpoints are untouched; "v2" adds action-history tokens, exact prize / deck-position
# knowledge and learned card-id embeddings (see NEXT_MODEL_DESIGN.md). Every process must
# agree, so main() and init_worker() both call set_encoding(). ---
ENCODING = "v1"
TOKEN_DIM = TOKEN_FEATURE_DIM_FULL_RICH                                # 340 + 16 = 356
GLOBAL_DIM = (GLOBAL_FEATURE_DIM_FULL + SELECT_CONTEXT_DIM
              + RICH_GLOBAL_DIM + SOLVER_GLOBAL_DIM)                   # 27+17+14+49 = 107
ZONE_COUNT = NUM_ZONES_FULL
MODEL_CARD_VOCAB = 0                                                   # 0 = no id embedding
OPTION_DIM = OPTION_FEATURE_DIM                                        # 908
MOVE_CAP = 1500


def set_encoding(name):
    """Point the module's dims at one encoding. Called before any model is built."""
    global ENCODING, TOKEN_DIM, GLOBAL_DIM, ZONE_COUNT, MODEL_CARD_VOCAB, OPTION_DIM
    ENCODING = name
    OPTION_DIM = OPTION_FEATURE_DIM
    if name == "v6":
        # v6 = v5 ++ the two chain-progress option columns (src/game/state_encoder.py): during
        # a BATCHED effect chain (Phantom Dive) the observation is frozen across the
        # placement selects, so the picks already accepted on each target and that target's
        # projected HP ride in as option features. Tokens / globals are v5's, unchanged.
        TOKEN_DIM = encode_inflight.TOKEN_FEATURE_DIM_V5
        GLOBAL_DIM = encode_inflight.GLOBAL_FEATURE_DIM_V5 + encode_selection.V4_GLOBAL_EXTRA_DIM
        ZONE_COUNT, MODEL_CARD_VOCAB = encode_inflight.NUM_ZONES_V5, V2_CARD_VOCAB
        OPTION_DIM = state_encoder.OPTION_FEATURE_DIM_V6
    elif name == "v5":
        # v5 = v4 ++ the 2026-07-30 audit fix tier (src/game/encode_inflight.py): revealed-prize
        # tokens, in-flight pick memory (+2 option columns, +2 globals), history serials,
        # mulligan / face-down-move events. Needs ActionHistory(extended=True) + an
        # InFlightTracker per seat. --encode-cache is v3-only and auto-disables here.
        TOKEN_DIM = encode_inflight.TOKEN_FEATURE_DIM_V5
        GLOBAL_DIM = encode_inflight.GLOBAL_FEATURE_DIM_V5 + encode_selection.V4_GLOBAL_EXTRA_DIM
        ZONE_COUNT, MODEL_CARD_VOCAB = encode_inflight.NUM_ZONES_V5, V2_CARD_VOCAB
        OPTION_DIM = encode_inflight.OPTION_FEATURE_DIM_V5
    elif name == "v4":
        # v4 = the v3 INPUTS ++ the model-decided multi-selection action surface
        # (src/game/encode_selection.py): +2 option columns (is_stop_action,
        # same_id_already_chosen) and +3 globals (picks_so_far, minCount, maxCount).
        TOKEN_DIM = TOKEN_FEATURE_DIM_V3
        GLOBAL_DIM = GLOBAL_FEATURE_DIM_V3 + encode_selection.V4_GLOBAL_EXTRA_DIM
        ZONE_COUNT, MODEL_CARD_VOCAB = NUM_ZONES_V3, V2_CARD_VOCAB
        OPTION_DIM = encode_selection.OPTION_FEATURE_DIM_V4
    elif name == "v3":
        # v2 ++ the audit-closing input surface (src/game/encode_details.py): de-aliased option
        # vectors (owner/area/slot + target instance state + SelectContext + attached-card
        # identity), face-down / deck-view tokens, absolute HP, wide copy counts, the three
        # dropped DumpState keys, a 224-event history window with target/attack identity.
        TOKEN_DIM, GLOBAL_DIM = TOKEN_FEATURE_DIM_V3, GLOBAL_FEATURE_DIM_V3
        ZONE_COUNT, MODEL_CARD_VOCAB = NUM_ZONES_V3, V2_CARD_VOCAB
        OPTION_DIM = OPTION_FEATURE_DIM + OPTION_EXTRA_DIM
    elif name == "v2":
        TOKEN_DIM, GLOBAL_DIM = TOKEN_FEATURE_DIM_V2, GLOBAL_FEATURE_DIM_V2
        ZONE_COUNT, MODEL_CARD_VOCAB = NUM_ZONES_V2, V2_CARD_VOCAB
    elif name == "v1":
        TOKEN_DIM = TOKEN_FEATURE_DIM_FULL_RICH
        GLOBAL_DIM = (GLOBAL_FEATURE_DIM_FULL + SELECT_CONTEXT_DIM
                      + RICH_GLOBAL_DIM + SOLVER_GLOBAL_DIM)
        ZONE_COUNT, MODEL_CARD_VOCAB = NUM_ZONES_FULL, 0
    else:
        raise ValueError(f"unknown encoding {name!r}")


def _has_trackers():
    """Encodings whose inputs include the per-seat CardKnowledge / ActionHistory trackers."""
    return ENCODING in ("v2", "v3", "v4", "v5", "v6")

# Aux heads (MY_MODEL_DESIGN.md, trained only when --aux-weight > 0): dense hindsight /
# hidden-info targets on the shared trunk, labels FREE from self-play (both perspectives
# run in one process). --heads family1 = the grounded-validated 4 (prize clocks k=1,
# survives, opp-hand flags); --heads full = the design-doc set (milestone prize clocks
# k=1..3 both sides, per-token KO clocks with evolution-line persistence, full-vocab
# opponent hand, deck-out clocks, hand-size trajectory, opponent's next supporter,
# opponent's next-turn active, survives).
NUM_TIMING_CLASSES = 9              # 0..6 own-side turns away; 7 = 7+; 8 = never
HAND_FLAGS = 5                      # basic pokemon / evolution / item / supporter / energy
KO_JOIN_MOVES = 4                   # left-board <-> prize proximity = a KO
MILESTONES = 6                      # prize clocks for the k-th next prize, k = 1..6
MAX_BOARD_TOKENS = 18               # 2 x (active + benchMax 8) -- per-token label padding
CARD_VOCAB = 1268                   # card ids are dense 1..1267; index 0 = none
HAND_SIZE_CAP = 15.0                # hand-size regression normalizer
DAMAGE_WINDOWS = 3                  # damage received within the next 1 / 2 / 3 own rounds
DAMAGE_CAP = 400.0                  # damage regression normalizer (max HP ~340)

# --- v2.1 head suite (NEXT_MODEL_DESIGN.md section 3, "The deferred head design").
# --heads v21 trains TokenFuture / SideFuture / ActionFuture (see --v21-modules). The
# legacy AuxHeads / AuxHeadsFull above are untouched: old checkpoints keep loading.
# Horizons are capped at +2 turns (spec): turn+1 is the OPPONENT's next turn and turn+2
# is OUR next turn, so a horizon index also fixes whose actions/state it describes.
V21_HORIZONS = 3                    # 0 = rest of THIS turn (mover), 1 = +1 (opp), 2 = +2 (us)
V21_PRIZE_K = MILESTONES            # turns_until_k_prizes, k = 1..6, both sides
ATTACK_VOCAB = 1557                 # attack ids are dense 1..1556; class 0 = "no attack"
ENERGY_CAP = 5.0                    # per-Pokemon attached-energy regression normalizer
SIDE_ENERGY_CAP = 12.0              # per-side total attached-energy normalizer
V21_BOARD_SLOTS = 4                 # board composition: (mine, theirs) x (+1, +2)
OPTION_TYPE_PLAY = 7                # cg.api.OptionType.* (ints inlined: no cg import here)
OPTION_TYPE_ATTACH = 8
OPTION_TYPE_EVOLVE = 9
OPTION_TYPE_ABILITY = 10
OPTION_TYPE_RETREAT = 12
OPTION_TYPE_ATTACK = 13
AREA_HAND = 2                       # cg.api.AreaType.HAND (PLAY's implied area)
LOG_TYPE_ATTACK = 15                # cg.api.LogType.ATTACK (playerIndex, cardId, attackId)
# Loss components, in the order their running-mean normalizers are stored (an aux_state
# buffer, so the order is part of the checkpoint format -- append, never reorder).
V21_COMPONENTS = ("ko_turns", "damage", "token_energy", "present",
                  "my_prize_turns", "opp_prize_turns", "deckout", "board", "side_energy",
                  "hand_sizes", "opp_hand", "opp_deck", "opp_active",
                  "plays", "abilities", "attack", "retreat", "my_use", "opp_use")

# --- Action payability (--attack-aux-weight, 0 = off and nothing below runs). Dense
# supervision for "which energies power up which attacks and abilities", so an attachment's
# usefulness reaches the trunk through something other than terminal win/loss noise.
#
# The labels are the ENGINE's OWN OFFERS at a MAIN select, never a cost computation here:
# whether an attack is payable given the attached energy (plus stadium cost modifiers,
# Tools, special energies, special conditions) -- and whether an ability is usable (energy
# requirement, once-per-turn already-used, effect locks) -- is EXACTLY "did the engine put
# it on the menu". Nothing in this file may decide payability itself.
#
# One board row carries PAYABLE_SLOTS columns: [attack 0, attack 1 | ability 0, ability 1],
# indexed by the card table's own `attacks` / `skills` order.
ATTACK_SLOTS = 2                 # a card carries at most 2 attacks (data/cards/cards.json)
SKILL_SLOTS = 2                  # ...and at most 2 skills (5 cards, all Antique Fossils)
PAYABLE_SLOTS = ATTACK_SLOTS + SKILL_SLOTS
PAYABLE_BOARD_SLOTS = 9          # the mover's own board tokens: active + benchMax 8
PAYABLE_HORIZON_SELECTS = 3      # horizon window = the mover's next 3 MAIN selects
SELECT_CONTEXT_MAIN = 0          # cg.api.SelectContext.MAIN -- the main action menu
AREA_ACTIVE = 4                  # cg.api.AreaType.ACTIVE / BENCH: an ABILITY option names
AREA_BENCH = 5                   # its HOST by (area, index), so those map it to a serial


# --- v2.2 head suite (SEARCH_TRAINING_DESIGN.md, "Aux head rework (v22 suite)"). --heads
# v22 is NEW code beside v21: the v21 classes and label paths are untouched and stay
# selectable, so every old checkpoint keeps loading.
#
# The horizon axis of the v21 ActionFuture heads (turn+h with a whose-turn lookup) is
# replaced by three DISJOINT windows anchored to the MOVER's own decision stream, so the
# actor of a window falls out of the window itself:
#   A  the mover's remaining actions this owned turn   (empty but VALID off-turn)
#   B  everything the OPPONENT does before the mover's next owned turn
#   C  the mover's next owned turn, start to finish
V22_WINDOWS = 3                     # A / B / C
V22_COUNT_CLASSES = 4               # per card id: {0, 1, 2, 3+}
V22_ATTACH_HORIZONS = 2             # attached cards at +1 / +2 turns
V22_STADIUM_HORIZONS = 2            # stadium in play at +1 / +2 turns
V22_LOCK_FLAGS = 5                  # item / supporter / stadium / special-energy / evolve
V22_LOCK_BITS = 2 * V22_LOCK_FLAGS  # ...for BOTH players (mine first, then theirs)
V22_MOVE_INFINITY = MOVE_CAP + 2    # "no upper bound": past any real move index
# Loss components, in the order their running-mean normalizers are stored (an aux_state
# buffer, so the order is part of the checkpoint format -- append, never reorder).
V22_COMPONENTS = ("ko_turns", "damage", "typed_attachments", "present",
                  "my_prize_turns", "opp_prize_turns", "deckout", "board",
                  "hand_sizes", "opp_hand", "opp_deck", "opp_active",
                  "my_prizes", "opp_prizes", "stadium", "future_locks",
                  "plays", "abilities", "attack", "retreat", "my_use", "opp_use")

# Which of those components each --heads value actually SUPERVISES. A name listed here
# keeps its head module, its label path and its checkpoint slot -- only the `add()` call is
# skipped, so every existing checkpoint still loads strictly and one restored line switches
# a channel back on. This is the pattern prizes_donated established on 2026-08-04, lifted
# out of the loss body so the enabled set is DATA rather than a comment, and so `--heads
# v22` stays bit-for-bit what it always was.
#   deckout    disabled 2026-08-05: acc 0.946 vs a majority-class base of 0.943.
#   my_prizes  disabled 2026-08-06 (v24 only): LABEL LEAK. The prize-pile id set the label
#              asks for is emitted straight back at the model as ZONE_PRIZE belief tokens
#              (encode_history.py:184) plus a global column asserting the deduction is exact
#              (encode_history.py:355). The trained head scores recall 0.972 on the rows where
#              that deduction is already exact and 0.073 where it is not -- it reads the
#              answer off its own input. Not specific to --prize-labels truth.
V22_DISABLED_BY_HEADS = {
    "v22": frozenset({"deckout"}),
    "v23": frozenset({"deckout"}),
    "v24": frozenset({"deckout", "my_prizes"}),
    #   plays      disabled 2026-08-06 (v25 only): 655 iters of d128_uniform put it at 16%
    #              of its honest condmaj bar (acc 0.151 vs "always predict the most common
    #              non-zero count" 0.93), stalled since mid-run. `abilities` stays: same
    #              family, but still climbing (0.52 -> 0.77 of its bar).
    #   board      disabled 2026-08-06 (v25 only): below the copy-current-state persistence
    #              baseline at BOTH model scales (d256 .868, d128 .844 vs ~.90-.92), flat
    #              300+ iters -- a head that cannot beat persistence supervises "copy the
    #              present", which the trunk gets for free.
    #   opp_prizes disabled 2026-08-06 (v25 only): flat since mid-run, ultra-conservative
    #              (fires on 15% of positive mass), and prize contents beyond deck-belief
    #              are hypergeometric arithmetic -- the same reason the v2.1 design
    #              rejected an opponent-prize head outright.
    "v25": frozenset({"deckout", "my_prizes", "plays", "board", "opp_prizes"}),
    #   typed_attachments disabled 2026-08-07 (v26): 700 iters of d256_uniform left it
    #              flat at acc ~0.68-0.69 since iter 100 while its honest condmaj bar
    #              (same non-zero cells) ROSE 0.83 -> 0.88 -- the gap is widening, not
    #              closing. A head below a degenerate predictor with zero trend dilutes
    #              the 16 that learn (the suite loss is a mean). `abilities` stays: it
    #              climbed 0.20 -> 0.56 with the bar-gap shrinking every 100-iter band.
    "v26": frozenset({"deckout", "my_prizes", "plays", "board", "opp_prizes",
                      "typed_attachments"}),
}

# v2.3 (AUX_V23_DESIGN.md) lives in aux_head_labels.py; it re-declares the two shapes its label
# arrays are cut to, so a change here can never silently un-align them from the heads.
assert aux_head_labels.V23_MAX_BOARD_TOKENS == MAX_BOARD_TOKENS
assert aux_head_labels.V23_ATTACK_SLOTS == ATTACK_SLOTS
assert aux_head_labels.V23_KO_JOIN_MOVES == KO_JOIN_MOVES
LOG_TYPE_DRAW, LOG_TYPE_MOVE_CARD = 4, 6      # cg.api.LogType.DRAW / MOVE_CARD
LOG_TYPE_HP_CHANGE = 16                       # ...HP_CHANGE (carries putDamageCounter)


def _attachment_vocabulary():
    """The ids a Pokemon can have ATTACHED to it: every energy card (basic + special) and
    every Pokemon Tool in the pool, read once from the card table the trainer already
    loads. Engine truth only -- an id is a column because a card with that id was observed
    attached, never because of what it "provides"."""
    from src.cards import CARDS
    ids = sorted(card_id for card_id, card in CARDS.items()
                 if card.get("cardType") in (2, 5, 6))     # TOOL / BASIC_ENERGY / SPECIAL
    return tuple(ids), {card_id: index for index, card_id in enumerate(ids)}


ATTACHMENT_IDS, ATTACHMENT_INDEX = _attachment_vocabulary()
ATTACHMENT_VOCAB = len(ATTACHMENT_IDS)                     # 47 today (27 tools + 20 energy)


def build_model(d_model=128, num_layers=3, num_heads=4, feedforward_dim=256):
    return GameStateTransformer(GameStateTransformerConfig(
        token_feature_dim=TOKEN_DIM, global_feature_dim=GLOBAL_DIM,
        option_feature_dim=OPTION_DIM, num_zones=ZONE_COUNT,
        d_model=d_model, num_layers=num_layers, num_heads=num_heads,
        feedforward_dim=feedforward_dim, card_vocab=MODEL_CARD_VOCAB))


def warm_start_v5_into_v6(state_dict, model):
    """A v5 checkpoint's state dict, made loadable into a v6 model by ZERO-PADDING the two
    new option columns onto the policy head's input projection.

    The head reads `cat([board context, option features])`, so v5 -> v6 widens exactly one
    parameter -- policy_score.0.weight -- by V6_OPTION_EXTRA_DIM input columns at the END
    (the chain block is appended last). Zeroing them makes the v6 forward EXACTLY the v5
    forward, so nothing the checkpoint learned is disturbed. Applied only for --encoding v6
    and only for that one parameter at exactly that one mismatch: every other shape
    disagreement is left for load_state_dict to reject loudly."""
    key = "policy_score.0.weight"
    loaded = state_dict.get(key)
    if ENCODING != "v6" or loaded is None:
        return state_dict
    expected = model.state_dict()[key]
    if (loaded.shape[0] != expected.shape[0]
            or expected.shape[1] - loaded.shape[1] != state_encoder.V6_OPTION_EXTRA_DIM):
        return state_dict
    padded = torch.zeros(expected.shape, dtype=loaded.dtype, device=loaded.device)
    padded[:, :loaded.shape[1]] = loaded
    print(f"warm start v5 -> v6: zero-padded {key} "
          f"{tuple(loaded.shape)} -> {tuple(padded.shape)}", flush=True)
    return dict(state_dict, **{key: padded})


def load_corpus_weighted():
    """-> (decks, weights): every 60-card corpus deck with its real ladder game count,
    so field sampling can mirror the actual metagame instead of uniform-over-1500."""
    records = json.loads((ROOT / "data" / "decks" / "corpus.json").read_text(encoding="utf-8"))
    decks, weights = [], []
    for record in records:
        deck = [int(card) for card, count in record["cards"].items()
                for _ in range(count)]
        if len(deck) == 60:
            decks.append(deck)
            weights.append(max(1, int(record.get("games", 1))))
    return decks, weights


def load_corpus_decks():
    return load_corpus_weighted()[0]


# ----------------------------------------------------------------------------------- #
# Worker side: persistent pool, CPU rollouts. Weights hot-reload by version number so
# the pool is never rebuilt; snapshots cache LRU-style (they are tiny, ~2.4 MB each).
# ----------------------------------------------------------------------------------- #

_worker = {}


# --- GPU-server message types ------------------------------------------------------- #
# A DECISION request is the 9-field payload + a request id, and its first field is the
# worker's integer index -- unchanged since the server was written. A SEARCH-LEAF BATCH is
# a second message type carrying many rows in ONE queue write, marked by a string in that
# same first slot (an int never equals it, so the discrimination is free and the decision
# path is byte-identical). Both types are drained, grouped, batched and forwarded by the
# same code; only the reply differs -- a leaf batch is answered ONCE, with the whole
# batch's priors and values as lists.
LEAF_TAG = "leaf"


def _message_rows(item):
    """How many model rows one queue message carries (1 for every decision request)."""
    return len(item[4]) if item[0] == LEAF_TAG else 1


def inference_server(request_queue, reply_queues, weights_dir, architecture,
                     encoding="v1", batch_cap=64, tf32=False):
    """GPU batcher process: drains worker requests, pads a batch, one CUDA forward,
    replies. Models load lazily per (kind, version) from the same weight files the
    workers use. fp32 inference for parity with the CPU path. Workers fall back to
    their local CPU model on any timeout, so this process can never wedge training.

    Assembly is whole-batch numpy scatters (the _collate_vectorized pattern) into REUSABLE
    PINNED staging buffers that stay at the wire dtypes -- f16 tokens/options, int16
    owners/zones, int32 card ids -- with the upcast to fp32/int64 done on the GPU. Measured
    2026-07-24: the per-row Python loop with astype() upcasts and fresh torch.zeros was
    5.12 ms of a 12.55 ms service cycle, and a pinned H2D is 0.56 vs 1.22 ms. The forward
    itself is unchanged fp32 (bf16 autocast measured SLOWER on this GPU)."""
    torch.set_num_threads(2)
    # SYMMETRY WITH THE TRAINING STEP (2026-08-08). This is a SPAWNED process, so it does
    # not inherit the parent's backend flags -- which is exactly why --tf32 failed its
    # 07-28 live test: generation ran fp32 while the training step ran TF32, so PPO's
    # ratio exp(new_logprob - old_logprob) straddled two numeric regimes and the gap
    # showed up as fake KL (kl_first_eval 0.002 -> 0.03, larger than real per-update
    # movement). Matching the server to the trainer removes the asymmetry; kl_first_eval
    # remains the canary that says whether it worked.
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    set_encoding(encoding)             # spawned process: dims must match the parent
    weights_dir = Path(weights_dir)
    models = {}                       # (kind, version) -> eval model on cuda

    def model_for(kind, version):
        key = (kind, version)
        if key not in models:
            name = f"current_v{version}.pt" if kind == "current" \
                else f"snapshot_v{version}.pt"
            state = torch.load(weights_dir / name, map_location="cpu",
                               weights_only=False)
            model = build_model(**(architecture or {}))
            model.load_state_dict(state)
            model.to("cuda").eval()
            stale = [k for k in models if k[0] == "current" and k != key]
            for k in stale[:-1]:      # keep at most 2 current versions + snapshots
                models.pop(k, None)
            models[key] = model
        return models[key]

    # Pinned staging, allocated once at a generous max shape and re-viewed per batch. The
    # buffers are FLAT so every per-batch view is contiguous: a strided slice of a padded
    # 3-D buffer would make torch stage an extra pageable copy and lose the pinned H2D.
    # A batch that overflows its reservation just reallocates that one buffer.
    staged_tokens = 256                                # padded token width reserved per row
    staged_options = 32                                # padded option width reserved per row
    staging = {
        "tokens": torch.empty(batch_cap * staged_tokens * TOKEN_DIM,
                              dtype=torch.float16).pin_memory(),
        "owners": torch.empty(batch_cap * staged_tokens, dtype=torch.int16).pin_memory(),
        "zones": torch.empty(batch_cap * staged_tokens, dtype=torch.int16).pin_memory(),
        "padding": torch.empty(batch_cap * staged_tokens, dtype=torch.bool).pin_memory(),
        "card_ids": torch.empty(batch_cap * staged_tokens, dtype=torch.int32).pin_memory(),
        "globals": torch.empty(batch_cap * GLOBAL_DIM, dtype=torch.float32).pin_memory(),
        "options": torch.empty(batch_cap * staged_options * OPTION_DIM,
                               dtype=torch.float16).pin_memory(),
        "option_mask": torch.empty(batch_cap * staged_options, dtype=torch.bool).pin_memory(),
    }

    def staged_view(name, shape):
        """A contiguous pinned view of `shape`, growing the reusable buffer if the batch
        needs more elements than the reservation."""
        buffer = staging[name]
        count = int(np.prod(shape))
        if buffer.numel() < count:
            buffer = torch.empty(count, dtype=buffer.dtype).pin_memory()
            staging[name] = buffer
        return buffer[:count].view(shape)

    # Per-stage service timings, printed every stats_window batches so the cycle time is
    # verifiable in production (a perf_counter pair per stage is ~100 ns).
    stats_window = 500
    stats = {name: 0.0 for name in ("batches", "requests", "drain", "assembly",
                                    "h2d", "forward", "reply")}

    while True:
        try:
            first = request_queue.get(timeout=1.0)
        except Exception:
            continue
        if first is None:
            return
        drain_start = time.perf_counter()
        batch = [first]
        # ROWS, not messages: a decision request is one row, so this loop is exactly the
        # message-counting one it replaces whenever no leaf batches are in flight.
        drained_rows = _message_rows(first)
        while drained_rows < batch_cap:
            try:
                item = request_queue.get_nowait()
            except Exception:
                break
            if item is None:
                return
            batch.append(item)
            drained_rows += _message_rows(item)
        stats["drain"] += time.perf_counter() - drain_start
        by_model = {}
        leaf_slots = {}                # (worker index, request id) -> [row result, ...]
        for item in batch:
            if item[0] == LEAF_TAG:
                _tag, index, kind, version, rows, sequence = item
                leaf_slots[(index, sequence)] = [None] * len(rows)
                for position, row in enumerate(rows):
                    # Leaf rows group SEPARATELY from decision rows (the third key field),
                    # so --leaf-fp16 can never pull a decision forward into fp16.
                    by_model.setdefault((kind, version, True), []).append(
                        (index, kind, version) + row + (sequence, position))
            else:
                by_model.setdefault((item[1], item[2]), []).append(item)
        for group_key, items in by_model.items():
            kind, version = group_key[0], group_key[1]
            try:
                model = model_for(kind, version)
                assembly_start = time.perf_counter()
                size = len(items)
                token_lengths = np.empty(size, dtype=np.int64)
                option_lengths = np.empty(size, dtype=np.int64)
                for row, item in enumerate(items):
                    token_lengths[row] = item[3].shape[0]
                    option_lengths[row] = item[7].shape[0]
                max_tokens, max_options = int(token_lengths.max()), int(option_lengths.max())
                # One CONTIGUOUS slice copy per request per field, plus a zero-fill of that
                # request's padded tail -- same bytes as the old whole-batch fancy scatter
                # (verified bit-identical) at ~3.4x, because a [row, :n] assignment is a
                # memcpy while `view[row_index, col_index] = concat(...)` walked every
                # element through numpy's advanced-indexing path AND built a full-batch
                # temporary first. Padded slots still end up zeroed, so no stale bytes from
                # the previous batch survive and the batch stays reproducible.
                # item[8] is the v2 per-token card-id array (None under v1).
                wants_ids = len(items[0]) > 8 and items[0][8] is not None
                tokens = staged_view("tokens", (size, max_tokens, items[0][3].shape[1]))
                token_view = tokens.numpy()
                owners = staged_view("owners", (size, max_tokens))
                owner_view = owners.numpy()
                zones = staged_view("zones", (size, max_tokens))
                zone_view = zones.numpy()
                globals_ = staged_view("globals", (size, items[0][6].shape[0]))
                global_view = globals_.numpy()
                options = staged_view("options", (size, max_options, items[0][7].shape[1]))
                option_view = options.numpy()
                card_ids = None
                card_view = None
                if wants_ids:
                    card_ids = staged_view("card_ids", (size, max_tokens))
                    card_view = card_ids.numpy()
                for row, item in enumerate(items):
                    count = int(token_lengths[row])
                    token_view[row, :count] = item[3]
                    owner_view[row, :count] = item[4]
                    zone_view[row, :count] = item[5]
                    if count < max_tokens:
                        token_view[row, count:] = 0
                        owner_view[row, count:] = 0
                        zone_view[row, count:] = 0
                    if card_view is not None:
                        card_view[row, :count] = item[8]
                        if count < max_tokens:
                            card_view[row, count:] = 0
                    count = int(option_lengths[row])
                    option_view[row, :count] = item[7]
                    if count < max_options:
                        option_view[row, count:] = 0
                    global_view[row] = item[6]
                padding = staged_view("padding", (size, max_tokens))
                padding.numpy()[:] = (np.arange(max_tokens)[None, :]
                                      >= token_lengths[:, None])
                option_mask = staged_view("option_mask", (size, max_options))
                option_mask.numpy()[:] = (np.arange(max_options)[None, :]
                                          < option_lengths[:, None])
                stats["assembly"] += time.perf_counter() - assembly_start
                with torch.no_grad():
                    h2d_start = time.perf_counter()
                    # f16/int16 across the bus, upcast on the GPU: the model still sees
                    # exactly the fp32 / int64 tensors the old CPU-side astype() produced.
                    device_tokens = tokens.to("cuda", non_blocking=True).float()
                    device_owners = owners.to("cuda", non_blocking=True).long()
                    device_zones = zones.to("cuda", non_blocking=True).long()
                    device_padding = padding.to("cuda", non_blocking=True)
                    device_globals = globals_.to("cuda", non_blocking=True)
                    device_options = options.to("cuda", non_blocking=True).float()
                    device_option_mask = option_mask.to("cuda", non_blocking=True)
                    device_card_ids = None if card_ids is None \
                        else card_ids.to("cuda", non_blocking=True).long()
                    torch.cuda.synchronize()
                    stats["h2d"] += time.perf_counter() - h2d_start
                    forward_start = time.perf_counter()
                    # --leaf-fp16: SEARCH-LEAF groups only (a leaf row carries the extra
                    # position field). Decision forwards stay fp32 -- their logprobs are
                    # stored in the training data, so their precision is not negotiable.
                    # fp16 leaf inference (design item 7) is CUT: GameStateTransformer's
                    # policy_value ends in `masked_fill(~option_mask, -1e9)`, and torch
                    # rejects that fill value on a half tensor whatever the mask holds --
                    # so an fp16 autocast around it raises, and the fix would have to
                    # change src/models/transformer.py, which this build may not touch.
                    # See check_search_gen.py [8]. Leaf forwards run fp32, like decisions.
                    logits, values = model.policy_value(
                        device_tokens, device_owners, device_zones, device_padding,
                        device_globals, device_options, device_option_mask,
                        card_ids=device_card_ids)
                    torch.cuda.synchronize()
                    stats["forward"] += time.perf_counter() - forward_start
                    reply_start = time.perf_counter()
                    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                    values = values.cpu().numpy()
                for row, item in enumerate(items):
                    count = item[7].shape[0]
                    answer = (probabilities[row, :count].astype(np.float32),
                              float(values[row]))
                    if len(item) > 10:          # a leaf row: answered with its batch below
                        leaf_slots[(item[0], item[9])][item[10]] = answer
                    else:
                        reply_queues[item[0]].put((item[9],) + answer)
                stats["reply"] += time.perf_counter() - reply_start
            except Exception:
                # Per (kind, version) group, so one bad group (a snapshot file pruned
                # under us) cannot kill the batch or the loop: those requests get an
                # empty reply and the worker falls back / retries. Leaf rows are left
                # unfilled and the flush below sends their batch's single error reply.
                for item in items:
                    if len(item) > 10:
                        continue
                    reply_queues[item[0]].put((item[9], None, None))
        for (index, sequence), slots in leaf_slots.items():
            # ONE reply per leaf batch: (request id, [priors...], [values...]), or the
            # (id, None, None) error shape the decision path already uses.
            if any(slot is None for slot in slots):
                reply_queues[index].put((sequence, None, None))
            else:
                reply_queues[index].put((sequence, [slot[0] for slot in slots],
                                         [slot[1] for slot in slots]))
        stats["batches"] += 1
        stats["requests"] += drained_rows
        if stats["batches"] >= stats_window:
            batches = stats["batches"]
            cycle = sum(stats[name] for name in ("drain", "assembly", "h2d",
                                                 "forward", "reply"))
            print(f"[gpu-server] {int(batches)} batches  "
                  f"mean size {stats['requests'] / batches:5.2f}  "
                  f"cycle {1000 * cycle / batches:6.2f} ms  (drain "
                  f"{1000 * stats['drain'] / batches:.2f}  assembly "
                  f"{1000 * stats['assembly'] / batches:.2f}  h2d "
                  f"{1000 * stats['h2d'] / batches:.2f}  forward "
                  f"{1000 * stats['forward'] / batches:.2f}  reply "
                  f"{1000 * stats['reply'] / batches:.2f})", flush=True)
            for name in stats:
                stats[name] = 0.0


def init_worker(weights_dir, deck_mode, probe_ckpt, architecture=None,
                server_queues=None, worker_counter=None, focus_deck=None,
                matchup_deck=None, encoding="v1", no_worker_model=False,
                opponent_bundle=None, probe_bundle=None, encode_cache=False,
                games_per_worker=1, resign_threshold=0.0, resign_persist=6,
                resign_audit_fraction=0.0, packed_transfer=True, payability=False,
                heads="family1", aux_labels=False,
                v21_modules=("token", "side", "action"), v21_checks=True,
                search=None, prize_labels="deduction", matchup_pool=None,
                field_pools=None):
    torch.set_num_threads(1)
    # --no-worker-model (oversubscribed pools): with the GPU server attached the local model
    # is ONLY the fallback forward, so at 40+ workers it is several GB of dead weight. Skip
    # it, and the server becomes mandatory (_policy_forward retries, then raises).
    assert not (no_worker_model and server_queues is None), \
        "--no-worker-model needs --gpu-server: no local model AND no server = no forward"
    set_encoding(encoding)             # dims must match the parent before any model is built
    _worker["no_model"] = bool(no_worker_model)
    _worker["server"] = None
    _worker["sequence"] = 0            # request id: the reply must carry it back (see below)
    # --encode-cache: byte-identical row-vectorised encoder (src/game/encode_history_cached.py),
    # v2 AND v3 layouts (the v3 row cache landed 2026-07-28, see V3_SPEED_REPORT.md).
    # --games-per-worker: how many games this worker keeps in flight at once, so its CPU work
    # overlaps the GPU server round trip instead of blocking on it.
    _worker["encode_cache"] = bool(encode_cache) and encoding in ("v2", "v3", "v4")
    _worker["concurrency"] = max(1, int(games_per_worker))
    # --packed-transfer: send the two big feature matrices of each decision to the parent
    # in their packed (nonzero-mask + values) form. Pure transport -- assemble unpacks to
    # byte-identical arrays -- but it is what keeps one K-game pipe write small enough for
    # Windows (the v3 widths hit ERROR_NO_SYSTEM_RESOURCES at 36 workers x K=4).
    _worker["packed_transfer"] = bool(packed_transfer)
    # --attack-aux-weight > 0: EventTracker also records what the engine offered at every
    # MAIN select (a handful of ints per select, see _capture_main_offers). Off = the
    # capture list stays empty and the record shipped to the parent is what it always was.
    _worker["payability"] = bool(payability)
    # --- LABEL BUILDING RUNS HERE (2026-07-31 pipeline rework). A finished game's aux labels
    # are a pure function of its own event/select record, so the worker that played it builds
    # them and the main process only concatenates -- the assemble phase used to be the
    # serial bottleneck. `v22` also switches on the extra per-decision engine-state capture
    # (attachment ids, stadium, turn ownership, prize residual, restriction bits).
    _worker["heads"] = heads
    _worker["aux_labels"] = bool(aux_labels)
    _worker["v21_modules"] = tuple(v21_modules)
    _worker["v21_checks"] = bool(v21_checks)
    # v23 is v22 PLUS the four v2.3 head groups, so it turns the v22 capture on as well.
    _worker["v22"] = bool(aux_labels) and heads in ("v22", "v23", "v24", "v25", "v26")
    _worker["v23"] = bool(aux_labels) and heads in ("v23", "v24", "v25", "v26")
    # --- search-in-the-loop generation. `search` is the whole SearchConfig-shaped dict or
    # None; None means not one line of the search path executes in this worker. ---
    _worker["search"] = search
    _worker["sessions"] = None
    _worker["recognizer"] = None
    _worker["truth_engine"] = None
    _worker["prize_labels"] = prize_labels
    # Resignation (see _selfplay_generator): config only -- every bit of PER-GAME state
    # lives in the generator's frame, never here, because one worker interleaves K games.
    _worker["resign_threshold"] = float(resign_threshold)
    _worker["resign_persist"] = max(1, int(resign_persist))
    _worker["resign_audit_fraction"] = float(resign_audit_fraction)
    if server_queues is not None:
        request_queue, reply_queues = server_queues
        with worker_counter.get_lock():
            index = worker_counter.value
            worker_counter.value += 1
        _worker["server"] = (request_queue, reply_queues[index], index)
    _worker["weights_dir"] = Path(weights_dir)
    _worker["deck_mode"] = deck_mode
    _worker["focus"] = focus_deck
    _worker["matchup"] = matchup_deck
    _worker["matchup_pool"] = matchup_pool
    # --field-pools: ((archetype name, (deck, ...)), ...) -- see _sample_archetype_deck.
    _worker["field_pools"] = field_pools
    _worker["corpus"], _worker["corpus_weights"] = load_corpus_weighted()
    alakazam_deck = ROOT / "decks" / "alakazam.csv"       # only --deck-mode alakazam reads it
    _worker["alakazam"] = ([int(line) for line in alakazam_deck.read_text().split()
                            if line.strip()] if alakazam_deck.exists() else None)
    _worker["opponents"] = {}
    _worker["snapshots"] = {}                          # version -> eval model
    # Bundle agent: a submission bundle's agent (e.g. the hand-built Alakazam
    # specialist), loaded once per worker via runpy (same mechanics as
    # experiments/alakazam_clone/eval_winrate.py; HYDRA_SEARCH=0 = scorer-only).
    # --opponent-bundle: seat 1 of GENERATION games is played by it (scripted mode).
    # --probe-bundle: loaded for the "specialist" PROBE only; generation is normal
    # self-play (e.g. matchup mode with the model on both seats).
    _worker["scripted"] = None
    _worker["scripted_deck"] = None
    _worker["bundle_agent"] = None
    _worker["bundle_deck"] = None
    bundle_path = opponent_bundle or probe_bundle
    if bundle_path:
        import runpy
        os.environ["HYDRA_SEARCH"] = "0"
        bundle = Path(bundle_path).resolve()       # MUST be absolute: we chdir into it
        _worker["bundle_deck"] = [int(line) for line in
                                  (bundle / "deck.csv").read_text().split()
                                  if line.strip()]
        cwd = os.getcwd()
        os.chdir(bundle)
        try:
            namespace = runpy.run_path(str(bundle / "main.py"),
                                       run_name="scripted_opponent")
        finally:
            os.chdir(cwd)
        _worker["bundle_agent"] = namespace["agent"]
        if opponent_bundle:
            _worker["scripted"] = _worker["bundle_agent"]
            _worker["scripted_deck"] = list(_worker["bundle_deck"])
    _worker["version"] = -1
    _worker["architecture"] = architecture or {}
    if _worker["no_model"]:
        _worker["model_eager"] = None      # every forward goes to the GPU server
        _worker["model"] = None
    else:
        model = build_model(**_worker["architecture"])
        model.eval()
        _worker["model_eager"] = model
        _worker["model"] = _deploy_model(model)
    _worker["solver"] = True
    if probe_ckpt:                                     # probe-only mode: any guide ckpt
        state = torch.load(probe_ckpt, map_location="cpu", weights_only=False)
        model = GameStateTransformer(GameStateTransformerConfig(
            token_feature_dim=state["token_feature_dim"],
            global_feature_dim=state["global_feature_dim"],
            option_feature_dim=state["option_feature_dim"],
            num_zones=state.get("num_zones", NUM_ZONES_FULL),
            d_model=state.get("d_model", 128),
            num_layers=state.get("num_layers", 3),
            num_heads=state.get("num_heads", 4),
            feedforward_dim=state.get("feedforward_dim", 256),
            card_vocab=state.get("card_vocab", 0),
            card_embedding_dim=state.get("card_embedding_dim", 64)))
        model.load_state_dict(state["state_dict"])
        model.eval()
        _worker["model_eager"] = model
        _worker["model"] = _deploy_model(model)
        _worker["solver"] = bool(state.get("solver_features", False))
        _worker["version"] = 0


def _deploy_model(eager_model):
    """The forward the worker actually calls: a TorchScript-frozen trace of policy_value
    (~1.9x on 1 CPU thread, outputs verified BITWISE-identical to eager across the shape
    distribution). Weights are baked in at trace time, so callers re-trace after every
    load_state_dict. Falls back to the eager module on any tracing failure (worst case =
    old speed, never wrong outputs). v1 models keep eager (their callers pass 7 args;
    the trace fixes an 8-arg signature)."""
    if not _has_trackers():
        return eager_model
    try:
        tokens = 24
        example = (torch.zeros(1, tokens, TOKEN_DIM), torch.zeros(1, tokens, dtype=torch.long),
                   torch.zeros(1, tokens, dtype=torch.long),
                   torch.zeros(1, tokens, dtype=torch.bool), torch.zeros(1, GLOBAL_DIM),
                   torch.zeros(1, 4, OPTION_DIM), torch.ones(1, 4, dtype=torch.bool),
                   torch.zeros(1, tokens, dtype=torch.long))
        with torch.no_grad():
            traced = torch.jit.trace_module(eager_model, {"policy_value": example})
            return torch.jit.freeze(traced, preserved_attrs=["policy_value"])
    except Exception as error:
        print(f"[worker] jit trace failed ({type(error).__name__}) -> eager forward",
              flush=True)
        return eager_model


def _refresh_weights(version):
    if _worker.get("no_model"):        # nothing local to refresh: the server holds the weights
        return
    if _worker["version"] == version:
        return
    state = torch.load(_worker["weights_dir"] / f"current_v{version}.pt",
                       map_location="cpu", weights_only=False)
    _worker["model_eager"].load_state_dict(state)
    _worker["model"] = _deploy_model(_worker["model_eager"])
    _worker["version"] = version


def _snapshot_model(version):
    cached = _worker["snapshots"].get(version)
    if cached is not None:
        return cached
    state = torch.load(_worker["weights_dir"] / f"snapshot_v{version}.pt",
                       map_location="cpu", weights_only=False)
    model = build_model(**_worker["architecture"])
    model.load_state_dict(state)
    model.eval()
    model = _deploy_model(model)
    if len(_worker["snapshots"]) >= 3:
        _worker["snapshots"].pop(min(_worker["snapshots"]))
    _worker["snapshots"][version] = model
    return model


def _sample_our_deck(deck_rng):
    """Seat 0 / the probe's seat: the deck WE pilot."""
    if _worker["deck_mode"] == "focus":
        return list(_worker["focus"])
    if _worker["deck_mode"] == "alakazam":
        return list(_worker["alakazam"])
    return deck_rng.choice(_worker["corpus"])


def _sample_archetype_deck(deck_rng):
    """--field-pools: a UNIFORM archetype, then a UNIFORM decklist inside it.

    Two independent uniform draws, deliberately NOT one uniform draw over all the lists:
    the pools are wildly uneven in size (alakazam 194 lists, lopunny 7), so flat-over-lists
    would hand ~25% of games to alakazam and ~1% to lopunny -- popularity weighting by the
    back door. Drawing the ARCHETYPE first makes every archetype an equally common
    opponent, which is the point: the model should meet each strategy often enough to learn
    it, not in proportion to how many players uploaded a list."""
    _name, pool = deck_rng.choice(_worker["field_pools"])
    return list(deck_rng.choice(pool))


def _sample_field_deck(deck_rng):
    """Seat 1 / opponents without their own deck: the metagame field. A fixed
    --matchup-deck overrides everything (matchup-expert training); --matchup-pool draws
    UNIFORMLY from one archetype's decklist pool (matchup fine-tuning -- uniform, not
    popularity-weighted, because popularity inside a pool is skewed enough that weighting
    would collapse it back to a single list); --field-pools draws uniform-archetype then
    uniform-list across ALL the pools (see _sample_archetype_deck); focus mode samples the
    corpus popularity-weighted; legacy modes mirror seat 0."""
    if _worker.get("scripted_deck"):
        return list(_worker["scripted_deck"])
    if _worker.get("matchup"):
        return list(_worker["matchup"])
    if _worker.get("matchup_pool"):
        return list(deck_rng.choice(_worker["matchup_pool"]))
    if _worker.get("field_pools"):
        return _sample_archetype_deck(deck_rng)
    if _worker["deck_mode"] == "focus":
        return list(deck_rng.choices(_worker["corpus"],
                                     weights=_worker["corpus_weights"], k=1)[0])
    return _sample_our_deck(deck_rng)


def _await_server_reply(reply_queue, sequence, timeout):
    """This request's reply, discarding any that belongs to an abandoned one. A request the
    worker gave up on can still be answered later, and an untagged reply would then be
    applied to a DIFFERENT decision (wrong state, silently) -- so the server echoes the
    request id and anything else is dropped. -> (probabilities, value), or None when the
    server reported an error; a timeout raises out of the queue get, as before."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        reply = reply_queue.get(timeout=remaining)
        if reply[0] == sequence:
            return None if reply[1] is None else (reply[1], reply[2])


def _encode_decision(observation, deck_counts, options, solver=True, knowledge=None,
                     history=None, encoder=None, in_flight=None, rich_out=None):
    """The LOCAL half of a forward: (encoded, option_features). Split out of
    _policy_forward so a pipelined worker can encode one game while another game's request
    is in flight -- the arrays it returns are exactly what _policy_forward built inline.

    `encoder`: an optional per-(game, seat) Cached{V2,V3}Encoder (--encode-cache). It
    returns byte-identical arrays to encode_observation_v2 / _v3 (src/game/
    encode_history_cached.py; certified by experiments/native_rollout) and is ~2.4x cheaper.

    v4 shares the v3 STATE and OPTION encoders: its own two option columns and three
    globals are per-SUB-PICK, so the v4 selection loop appends them to what this returns
    (src/game/encode_selection.py).

    `rich_out`: a list the caller passes to receive this decision's decoded engine state
    (the (cards, players) pair `encode_rich.rich_state` returns). The decode is then done
    HERE and handed to the encoder as `rich_override`, which is exactly what the encoder
    would have decoded itself -- so the v2.2 labels that read the engine state get it
    WITHOUT a second dump_state call per decision."""
    rich_override = None
    if rich_out is not None:
        # rich_out[0] = the (cards, players) pair, exactly as before; rich_out[1] = the
        # TOP-LEVEL dump fields that pair drops (build_v25 `knockouts`).
        extras = {}
        rich_override = rich_state(observation, extras_out=extras)
        rich_out.append(rich_override)
        rich_out.append(extras)
    if ENCODING in ("v5", "v6"):
        # No cached path yet: encode_history_cached is certified against v3 arrays only.
        # v6's state encoder IS v5's (state_encoder.encode_observation_v6): only options widen.
        encoded = encode_inflight.encode_observation_v5(observation, deck_counts=deck_counts,
                                                  knowledge=knowledge, history=history,
                                                  in_flight=in_flight,
                                                  rich_override=rich_override)
    elif ENCODING in ("v3", "v4"):
        if encoder is not None:
            encoded = encoder.encode(observation, deck_counts=deck_counts,
                                     knowledge=knowledge, history=history)
        else:
            encoded = encode_observation_v3(observation, deck_counts=deck_counts,
                                            knowledge=knowledge, history=history,
                                            rich_override=rich_override)
    elif ENCODING == "v2":
        # v2 has no solver block (owner call 2026-07-23): factual inputs only.
        if encoder is not None:
            encoded = encoder.encode(observation, deck_counts=deck_counts,
                                     knowledge=knowledge, history=history)
        else:
            encoded = encode_observation_v2(observation, deck_counts=deck_counts,
                                            knowledge=knowledge, history=history,
                                            rich_override=rich_override)
    else:
        encoded = encode_observation_full(observation, deck_counts=deck_counts,
                                          include_rich=True, include_select_context=True,
                                          include_solver_features=solver,
                                          rich_override=rich_override)
    select = observation.get("select") or {}
    encode_one = encode_option_v3 if ENCODING in ("v3", "v4", "v5", "v6") else \
        (lambda obs, option, _select: encode_option(obs, option))
    option_features = np.stack([encode_one(observation, option, select)
                                for option in options]).astype(np.float32)
    return encoded, option_features


def _model_row(encoded, option_features):
    """The six model-input arrays of ONE row, at the wire dtypes."""
    card_ids = encoded.get("card_ids")
    return (encoded["token_features"].astype(np.float16),
            encoded["owner_ids"].astype(np.int16),
            encoded["zone_ids"].astype(np.int16),
            encoded["global_features"].astype(np.float32),
            option_features.astype(np.float16),
            None if card_ids is None else card_ids.astype(np.int32))


def _server_payload(index, tag, encoded, option_features):
    """The wire tuple the GPU server consumes (minus the trailing request id)."""
    return (index, tag[0], tag[1]) + _model_row(encoded, option_features)


# ----------------------------------------------------------------------------------- #
# Search-leaf encoding (--search-gen). The tree in search_gen.py treats these payloads as
# opaque; everything encoding-specific lives here, next to _encode_decision, so the two
# cannot drift.
# ----------------------------------------------------------------------------------- #

_ZERO_OPTION_ROW = None            # lazily sized: the value-only request's dummy option


def _search_state_key(observation):
    """The transposition key: a search node's SERIAL LAYOUT plus the menu it is offering.

    Attach / play / bench orderings transpose constantly, so two different action orders
    inside one turn reach byte-identical states with different search ids. Keying on the
    physical cards (serials, hp, what is attached to what, hand and discard membership,
    counts) plus the offered option tuples makes those one cache entry -- and makes a
    collision impossible for any state the encoder would have encoded differently."""
    current = observation.get("current")
    if current is None:
        return None
    select = observation.get("select") or {}
    parts = [current.get("yourIndex"), current.get("turn"), current.get("turnActionCount"),
             current.get("result"), bool(current.get("supporterPlayed")),
             bool(current.get("stadiumPlayed")), bool(current.get("energyAttached")),
             bool(current.get("retreated")),
             (select.get("context"), select.get("minCount"), select.get("maxCount"))]
    for player in current.get("players") or []:
        board = []
        for pokemon in ((player.get("active") or []) + (player.get("bench") or [])):
            if pokemon is None:
                board.append(0)
                continue
            board.append((pokemon.get("serial"), pokemon.get("hp"),
                          tuple(card["serial"] for card in
                                (pokemon.get("energyCards") or [])),
                          tuple(card["serial"] for card in (pokemon.get("tools") or [])),
                          tuple(card["serial"] for card in
                                (pokemon.get("preEvolution") or []))))
        parts.append((tuple(board), player.get("deckCount"), player.get("handCount"),
                      tuple(card["serial"] for card in (player.get("hand") or [])),
                      tuple(card["serial"] for card in (player.get("discard") or [])),
                      len(player.get("prize") or []),
                      player.get("poisoned"), player.get("burned"), player.get("asleep"),
                      player.get("paralyzed"), player.get("confused")))
    parts.append(tuple(card["serial"] for card in (current.get("stadium") or [])))
    parts.append(tuple((option.get("type"), option.get("area"), option.get("index"),
                        option.get("playerIndex"), option.get("attackId"),
                        option.get("cardId"), option.get("number"),
                        option.get("toolIndex"), option.get("energyIndex"),
                        option.get("inPlayArea"), option.get("inPlayIndex"))
                       for option in (select.get("option") or [])))
    return tuple(parts)


def _rich_from_dump(dump):
    """A DumpSearchState blob in the shape encode_rich.rich_state returns."""
    return ({card["serial"]: card for card in dump.get("cards") or []},
            {player["playerIndex"]: player for player in dump.get("players") or []})


class SearchLeafEncoder:
    """Turns a live search node into model input, with the per-decision transposition and
    encode cache (design item 4).

    ROOT-FROZEN CONTEXT, the convention the shipped search wrapper already uses
    (experiments/ladder_bc/search_agent.py::V2EncoderGuide): a search node is a
    hypothetical and cannot produce this seat's CardKnowledge / ActionHistory /
    InFlightTracker itself, so the deciding seat's trackers are captured once at the
    real decision and applied to every node. The rich EFFECT block is NOT frozen --
    `DumpSearchState` gives every leaf its own live one (addendum 13).

    `lookup`/`store` are the only surface `search_gen` sees.
    """

    __slots__ = ("session", "deck_counts", "knowledge", "history", "in_flight", "cache",
                 "states", "encodes", "option_encodes", "dump_failures")

    def __init__(self, session, deck_counts, knowledge, history, in_flight=None):
        self.session = session
        self.deck_counts = deck_counts
        self.knowledge = knowledge
        self.history = history
        # Deepcopied so simulation can never mutate real play state -- the same guarantee
        # the deployed wrapper makes (submission main.py passes copy.deepcopy into every
        # in-search resolve). Leaf encodes only READ it, but the copy keeps the frozen
        # contract independent of that detail.
        self.in_flight = copy.deepcopy(in_flight)
        self.cache = {}                # state key -> {"value":..., "priors":...}
        self.states = {}               # search id -> (key, observation, encoded)
        self.encodes = 0
        self.option_encodes = 0
        self.dump_failures = 0

    def _state(self, search_id):
        """(key, observation, encoded state arrays) for a node, encoded at most once."""
        hit = self.states.get(search_id)
        if hit is not None:
            return hit
        import dataclasses
        observation = dataclasses.asdict(self.session.observation(search_id))
        key = _search_state_key(observation)
        cached = self.cache.get(key)
        encoded = None if cached is None else cached.get("encoded")
        if encoded is None:
            rich = None
            try:
                rich = _rich_from_dump(self.session.dump(search_id))
            except Exception:
                self.dump_failures += 1
            self.encodes += 1
            encoded = encode_inflight.encode_observation_v5(
                observation, deck_counts=self.deck_counts, knowledge=self.knowledge,
                history=self.history, in_flight=self.in_flight, rich_override=rich)
            self.cache.setdefault(key, {})["encoded"] = encoded
        entry = (key, observation, encoded)
        self.states[search_id] = entry
        return entry

    def _leaf_globals(self, key, encoded, select):
        """The v3/v5 globals ++ the three v4 sub-pick columns, at the START of a select
        (no picks taken). Identical to `encode_selection.global_features(base, MultiSelect(...))`
        with an empty `chosen`, without paying for the MultiSelect -- and used by BOTH the
        value-only and the priors request, so a node's state vector is the same array
        whichever request happens first (this is what makes lazy and eager expansion
        bit-identical, and it is also the model's real global width)."""
        cached = self.cache.setdefault(key, {}).get("globals")
        if cached is not None:
            return cached
        low, high = encode_selection.take_bounds(select)
        extra = np.zeros(encode_selection.V4_GLOBAL_EXTRA_DIM, dtype=np.float32)
        extra[encode_selection.G_PICKS_SO_FAR] = 0.0
        extra[encode_selection.G_MIN_COUNT] = min(low, encode_selection.PICK_SCALE) \
            / encode_selection.PICK_SCALE
        extra[encode_selection.G_MAX_COUNT] = min(high, encode_selection.PICK_SCALE) \
            / encode_selection.PICK_SCALE
        globals_ = np.concatenate([encoded["global_features"], extra]).astype(np.float32)
        self.cache[key]["globals"] = globals_
        return globals_

    def _rows(self, key, observation, encoded, with_options):
        """The wire row. `with_options=False` is the VALUE-ONLY request of lazy expansion:
        the value head reads the trunk context alone (transformer.policy_value), so a
        single zero option row gives the identical value at none of the option-encoding
        cost -- and its logits are simply never read."""
        global _ZERO_OPTION_ROW
        select = observation.get("select") or {}
        step = dict(encoded, global_features=self._leaf_globals(key, encoded, select))
        if not with_options:
            if _ZERO_OPTION_ROW is None or _ZERO_OPTION_ROW.shape[1] != OPTION_DIM:
                _ZERO_OPTION_ROW = np.zeros((1, OPTION_DIM), dtype=np.float32)
            return _model_row(step, _ZERO_OPTION_ROW)
        # The option matrix of a single-pick select, built exactly as the generator builds
        # it for a real decision, so a leaf and a decision are encoded alike.
        self.option_encodes += 1
        state = encode_selection.MultiSelect(observation, select)
        if ENCODING == "v6":
            v3_matrix, option_extra = state_encoder.base_option_matrix_v6(observation, select,
                                                                      self.in_flight)
        else:
            v3_matrix, option_extra = encode_inflight.base_option_matrix_v5(observation, select,
                                                                      self.in_flight)
        # stop_offered is forced FALSE: the tree's edge set is the engine's option list, so
        # a synthetic STOP row would hand back one prior more than there are edges. Search
        # nodes with minCount 0 therefore score their options without the decline
        # candidate the real v5 decision loop would also offer -- a prior-side
        # approximation at INTERNAL nodes only (values are unaffected), flagged in the
        # build report. Searched ROOT decisions are restricted to minCount == 1, so the
        # emitted target is always indexed exactly like the recorded option rows.
        if ENCODING == "v6":
            rows = state_encoder.candidate_matrix_v6(v3_matrix, option_extra, state,
                                                 state.pending(), False)
        else:
            rows = encode_inflight.candidate_matrix_v5(v3_matrix, option_extra, state,
                                                 state.pending(), False)
        return _model_row(step, rows)

    def request(self, search_id, kind):
        """search_gen's contract: ("hit", priors, value) | ("miss", payload, key) | None."""
        try:
            key, observation, encoded = self._state(search_id)
        except Exception:
            return None
        select = observation.get("select") or {}
        if not (select.get("option") or []):
            return None
        entry = self.cache.setdefault(key, {})
        if kind == "value" and "value" in entry:
            return "hit", entry.get("priors"), entry["value"]
        if kind == "priors" and "priors" in entry:
            return "hit", entry["priors"], entry.get("value")
        if kind == "both" and "value" in entry and "priors" in entry:
            return "hit", entry["priors"], entry["value"]
        try:
            row = self._rows(key, observation, encoded, kind != "value")
        except Exception:
            return None
        return "miss", row, key

    def store(self, key, kind, priors, value):
        entry = self.cache.setdefault(key, {})
        if kind in ("value", "both"):
            entry["value"] = float(value)
        if kind in ("priors", "both") and priors is not None:
            entry["priors"] = list(priors)


# --- worker -> parent transport ---------------------------------------------------- #
# A generation block's per-decision feature matrices ARE its pipe payload, and they are
# mostly zeros (one-hots, unused blocks, whole zero slices where a row kind carries no
# state). Pickling them dense is what blew up as Windows ERROR_NO_SYSTEM_RESOURCES (1450)
# on the fatter v3 widths. `pack_dense` is LOSSLESS AND BITWISE: the zero test runs on the
# raw bits, so -0.0 is stored as a value rather than folded into the implicit zeros, and
# `unpack_dense` reproduces the worker's array byte for byte (dtype, shape and bytes).
_PACK_UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def pack_dense(array):
    """array -> (shape, dtype, packed nonzero mask, nonzero values)."""
    flat = np.ascontiguousarray(array).reshape(-1)
    nonzero = flat.view(_PACK_UINT[array.dtype.itemsize]) != 0
    return (array.shape, array.dtype.str, np.packbits(nonzero), flat[nonzero])


def _nonzero_offsets(bits, size):
    """Flat positions of the packed mask's set bits.

    `mask = unpackbits(...).view(bool)` then `destination[mask] = values` walks all `size`
    elements through numpy's boolean-mapping loop; `destination[flatnonzero(mask)] = values`
    touches only the ~11 % that are set and is **3.5x cheaper** on the real shapes (0.095 ->
    0.027 ms for one v3 token matrix). Same positions, same order, same bytes -- the
    boolean form is literally defined as this index set. `flatnonzero` must see a BOOL
    array: on the raw uint8 from `unpackbits` it loses the fast path and is slower than the
    mask assign it replaces (0.111 ms)."""
    return np.flatnonzero(np.unpackbits(bits, count=size).view(bool))


def unpack_dense(packed):
    """The inverse of pack_dense; byte-identical to the array that went in."""
    shape, dtype, bits, values = packed
    size = int(np.prod(shape, dtype=np.int64))
    restored = np.zeros(size, dtype=dtype)
    restored[_nonzero_offsets(bits, size)] = values
    return restored.reshape(shape)


def _maybe_unpack(value):
    """A decision's feature matrix, whether the worker sent it dense or packed. The form is
    self-describing (ndarray vs pack tuple), so --no-packed-transfer records and packed
    records assemble through the same code and old harnesses keep working."""
    return value if isinstance(value, np.ndarray) else unpack_dense(value)


def dense_shape(value):
    """The shape the decision's matrix HAS or WOULD have -- readable without unpacking."""
    return value.shape if isinstance(value, np.ndarray) else value[0]


def scatter_dense(value, destination):
    """Write one decision's feature matrix into `destination`, a view of the collated
    batch's padded block that is already zero.

    This is the whole point of LAZY_UNPACK: the packed form goes straight into the batch,
    so the intermediate dense array is never allocated, never zero-filled and never copied
    a second time. Byte-identical to `destination[...] = unpack_dense(value)` -- the
    implicit zeros are the zeros the destination was allocated with, and the set positions
    get the same values in the same order."""
    if isinstance(value, np.ndarray):
        destination[...] = value
        return
    if not destination.flags.c_contiguous:
        # reshape(-1) would COPY and the scatter would be silently thrown away.
        raise ValueError("scatter_dense destination must be C-contiguous")
    shape, dtype, bits, values = value
    size = int(np.prod(shape, dtype=np.int64))
    destination.reshape(-1)[_nonzero_offsets(bits, size)] = values


# --parent-side unpack placement --------------------------------------------------------- #
# The worker packs (above); somebody on the parent has to put the bytes back. Doing it in
# `assemble` cost 0.18-0.29 ms/decision = 23-35 s/iteration of SERIAL allocate-and-fill on
# the critical path between generation and the GPU step (V3_SPEED_REPORT.md R6). With
# LAZY_UNPACK the packed form is carried through `assemble` untouched and scattered ONCE,
# directly into the padded minibatch, inside `collate` -- which had to copy those bytes
# anyway, so the intermediate array's allocation, zero-fill and copy all disappear.
# --eager-unpack restores the old placement (byte-identical, just slower).
LAZY_UNPACK = True


@torch.no_grad()
def _local_forward(model, encoded, option_features):
    """The worker's own CPU forward -> (probabilities [O], value)."""
    card_ids = encoded.get("card_ids")
    tokens = torch.from_numpy(encoded["token_features"]).unsqueeze(0)
    owners = torch.from_numpy(encoded["owner_ids"]).unsqueeze(0)
    zones = torch.from_numpy(encoded["zone_ids"]).unsqueeze(0)
    padding = torch.zeros(1, tokens.shape[1], dtype=torch.bool)
    globals_ = torch.from_numpy(encoded["global_features"]).unsqueeze(0)
    option_tensor = torch.from_numpy(option_features).unsqueeze(0)
    option_mask = torch.ones(1, option_features.shape[0], dtype=torch.bool)
    identity = (torch.from_numpy(card_ids.astype(np.int64)).unsqueeze(0)
                if card_ids is not None else None)
    logits, value = model.policy_value(tokens, owners, zones, padding, globals_,
                                       option_tensor, option_mask, card_ids=identity)
    return torch.softmax(logits[0], dim=-1).numpy(), float(value[0])


@torch.no_grad()
def _policy_forward(model, observation, deck_counts, options, solver=True, tag=None,
                    knowledge=None, history=None, encoder=None):
    """One forward: (probabilities [O], value, encoded, option_features). Encoding always
    happens locally (matches FullEncoderGuide); the net runs on the GPU server when one
    is attached and a model tag is given, with local-CPU fallback on any hiccup.

    v2 additionally takes this seat's OWN trackers -- `knowledge` (prize / deck-position
    deduction) and `history` (visible action stream). Passing another seat's trackers would
    be an information leak, so play_selfplay_game keeps one set per seat and never shares."""
    encoded, option_features = _encode_decision(observation, deck_counts, options,
                                                solver=solver, knowledge=knowledge,
                                                history=history, encoder=encoder)
    probabilities, value = _serve_forward(model, tag, encoded, option_features)
    return probabilities, value, encoded, option_features


@torch.no_grad()
def _serve_forward(model, tag, encoded, option_features):
    """(probabilities [O], value) for already-encoded inputs: the GPU server when one is
    attached and a model tag is given, local CPU otherwise. Lifted out of _policy_forward
    unchanged so the v4 selection loop can serve a sub-pick whose option matrix it built
    itself."""
    if _worker.get("server") is not None and tag is not None:
        request_queue, reply_queue, index = _worker["server"]
        payload = _server_payload(index, tag, encoded, option_features)
        try:
            sequence = _worker["sequence"] = _worker["sequence"] + 1
            request_queue.put(payload + (sequence,))
            reply = _await_server_reply(reply_queue, sequence, 30.0)
            if reply is not None:
                return reply[0], reply[1]
        except Exception:
            pass                                       # fall through to local CPU
        if _worker["no_model"]:
            # No local fallback exists, and answering with a random move would poison the
            # training data invisibly. Re-send the SAME payload up to twice more (a fresh
            # request id each time), then give up loudly: the caller's per-decision handler
            # counts it in the iter line's `errors`, so a broken server shows up instead of
            # hiding in the data.
            for _attempt in range(2):
                try:
                    sequence = _worker["sequence"] = _worker["sequence"] + 1
                    request_queue.put(payload + (sequence,))
                    reply = _await_server_reply(reply_queue, sequence, 5.0)
                    if reply is not None:
                        return reply[0], reply[1]
                except Exception:
                    pass
            raise RuntimeError("server unavailable")
    return _local_forward(model, encoded, option_features)


def _sample_index(probabilities, rng):
    roll = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += float(probability)
        if roll < cumulative:
            return index
    return len(probabilities) - 1


def _trivial_move(select):
    """The non-learnable selects (same handling as SearchBCAgent): no options, take-all,
    or a multi-pick. Returns None when this IS a learnable single-choice decision.

    v1/v2/v3 ONLY. Three of its four branches answer prompts that are real decisions (which
    cards to discard, which two to search out, whether to use an ability at all) -- see
    src/game/encode_selection.py. --encoding v4 never calls this: it uses
    encode_selection.forced_answer (single-legal-answer prompts only) + the selection loop."""
    options = select["option"]
    count = len(options)
    if count == 0:
        return []
    if select["maxCount"] >= count:
        return list(range(count))
    if select["maxCount"] != 1:
        return list(range(max(min(select["maxCount"], count), select["minCount"])))
    if count == 1:
        return [0]
    return None


def _random_legal(select, rng):
    count = len(select["option"])
    take = max(min(select["maxCount"], count), select["minCount"])
    return sorted(rng.sample(range(count), take)) if count else []


def _first_active_serial(player):
    active = next((p for p in (player.get("active") or []) if p is not None), None)
    return active["serial"] if active else None


def _board_serials(player):
    """Every serial on this player's board incl. evolution stacks (an evolve or retreat
    must never look like the active vanishing)."""
    serials = set()
    for pokemon in (player.get("active") or []) + (player.get("bench") or []):
        if pokemon is None:
            continue
        serials.add(pokemon["serial"])
        for underneath in (pokemon.get("preEvolution") or []):
            if isinstance(underneath, dict) and "serial" in underneath:
                serials.add(underneath["serial"])
    return serials


def _discard_supporters(player):
    from src.cards import get_card
    counts = {}
    for card in player.get("discard") or []:
        card_data = get_card(card["id"]) or {}
        if card_data.get("cardType") == 3:
            counts[card["id"]] = counts.get(card["id"], 0) + 1
    return counts


def _option_card_id(player, option, default_area=None):
    """Card id an option points at: its area + index within that area. PLAY carries only
    an index (into the hand), so callers pass the area it implies."""
    area = option.get("area") or default_area
    index = option.get("index")
    if index is None:
        return None
    holder = {4: player.get("active"), 5: player.get("bench"), 2: player.get("hand"),
              3: player.get("discard")}.get(area)
    if not holder or not (0 <= index < len(holder)) or holder[index] is None:
        return None
    return holder[index].get("id")


def _v22_prize_ids(rich_cards, mover, deck_counter, prize_count):
    """The card ids sitting in MOVER's prize pile at this decision, or None (not deducible
    from THIS decision -- see _v22_prize_eras, which propagates a deduction across the
    interval where the pile provably did not change).

    The engine's state blob is the DECIDING seat's own KNOWLEDGE state, and DumpState emits
    every card in it whose id is known. A prize card's id is exactly what its owner does not
    know, so prize rows never appear at all (probed: area 6 is absent from every dump, both
    seats) -- the spec's "extract the prize-area ids" is not available from this engine.

    What IS available: the prizes are the RESIDUAL of the 60-card decklist after every card
    the dump does place, and that residual collapses to the prize pile exactly when the
    seat's own DECK is enumerated too -- which the engine does whenever that seat is looking
    at its deck (measured: 91% of the deck-enumerated dumps are deck-search selects). The
    check that closes the derivation is the engine's own prize count: a residual of the
    wrong size still holds unseen deck cards, so it is refused rather than guessed."""
    if not rich_cards:
        return None
    placed = Counter()
    for card in rich_cards.values():
        if card.get("playerIndex") == mover and card.get("cardId"):
            placed[card["cardId"]] += 1
    residual = deck_counter - placed              # Counter: negatives clamp to absent
    ids = sorted(residual.elements())
    return tuple(ids) if len(ids) == prize_count else None


def _v22_lock_bits(rich_players, mover):
    """The 10 restriction bits (mine, then theirs) of the 5 per-player locks -- item /
    supporter / stadium / special-energy / evolve. Same expressions as the RICH INPUT block
    the model already sees (encode_rich.player_block's first five columns), so the label and
    the feature are the same engine fact. None (-> masked) when the dump is unavailable."""
    if not rich_players:
        return None
    return np.concatenate([player_block(rich_players.get(mover))[:V22_LOCK_FLAGS],
                           player_block(rich_players.get(1 - mover))[:V22_LOCK_FLAGS]]
                          ).astype(np.float32)


class EventTracker:
    """Both-sides hindsight event stream over the self-play observation flow (the board is
    public; each side's hand is visible on its own observations): prize takes, EVERY board
    departure per serial (evolution-line safe -- an evolving serial stays inside the stack's
    _board_serials, so its clock runs until the whole stack leaves), deck-empty, per-turn
    snapshots (hand counts + discard supporter counts), plus true-hand / next-active capture
    for pending decisions.

    v2.1 additions (all additive and read-only -- they never touch the engine or an rng, so
    trajectories are bit-identical): ATTACK log events, ABILITY/RETREAT events from BOTH
    seats' selection streams (`record_selection`), per-turn board composition / attached
    energy / presence in the snapshots, start-of-turn hand sizes, and each seat's own hand
    contents over time (its hand is fully visible on its own observations)."""

    def __init__(self, capture_main_offers=False, capture_v22=False, capture_v23=False):
        self.events = []
        self.snapshots = []               # one per completed turn: hands + supporter counts
        self.pending = {0: [], 1: []}     # mover -> metas awaiting opponent's next-turn view
        self.pending_turn = {0: -1, 1: -1}
        self.prev_prizes = None
        self.prev_board = [set(), set()]
        self.prev_tops = [{}, {}]         # top serial -> (hp, stack serial list, maxHp)
        self.prev_serial_hp = [{}, {}]    # any stack serial -> its stack top's last hp
        self.prev_turn = None
        self.last_hands = [0, 0]
        self.deck_empty = [None, None]    # (move, turn) when deckCount first hits 0
        # --- v2.1 label inputs ---
        self.turn_start_hands = {}        # turn -> [hand count 0, hand count 1] at its start
        self.hand_views = [[], []]        # seat -> [(move, (card ids,))], only on CHANGE
        self.last_board_ids = [[], []]    # in-play card ids per side (top of each stack)
        self.last_active_ids = [None, None]   # each side's ACTIVE card id (None if empty)
        self.last_energy = {}             # any stack serial -> attached energy count
        self.last_energy_total = [0, 0]   # attached energy summed over a side's board
        self.last_present = set()         # every serial currently in play (both sides)
        # --- payability capture (--attack-aux-weight; empty and never written to when the
        # flag is off, so the record shipped to the parent is unchanged) ---
        self.capture_main_offers = capture_main_offers
        self.main_offers = []             # one entry per MAIN select, see record_selection
        # --- v2.2 capture (--heads v22; empty and never written to when off) ---
        self.capture_v22 = capture_v22
        self.last_attachments = {}        # any stack serial -> the ids ATTACHED to its top
        # --- v23 capture: per-energy IDENTITY (serials), keyed by the stack's BOTTOM
        # serial (stable across evolution, unlike the top). Feeds attach_seen /
        # energy_lost events for the attachment-need labels. ---
        self.prev_energy_serials = {}     # bottom serial -> (owner, ((serial, id), ...))
        self.last_stadium = 0             # card id of the stadium in play (0 = none)
        self.turn_first_move = {}         # turn -> the first move index it was observed at
        self.turn_owner = {}              # turn -> the seat the engine offers its MAIN menu
        self.decision_facts = {0: [], 1: []}   # seat -> per-select engine-state facts
        # --- v2.3 capture (--heads v23; empty and never written to when off) ---
        self.capture_v23 = capture_v23
        self.v23_energy_state = []        # (move, mover, {holder -> engine energy facts})
        # The engine hands each SEAT the logs since ITS OWN last observation (State::
        # nextLogStart), so the two seats' streams each cover every public event exactly
        # once. Counting what each seat has consumed turns its stream into ABSOLUTE log
        # indices, and keeping only indices not seen before dedupes the two streams
        # exactly -- no tuple matching, no heuristics.
        self.log_seen = [0, 0]
        self.log_high = 0
        self.prev_bottoms = {}            # in-play serial -> its stack's bottom serial
        self.ko_count = 0                 # engine KO records consumed (build_v25 dump)
        self.ko_scan_high = -1            # last move index a knockout scan covered

    def observe(self, observation, move_index):
        current = observation["current"]
        players = current["players"]
        turn = current["turn"]
        # Card USE events straight from the engine log: PLAY(10) / ATTACH(11) / EVOLVE(12)
        # -- deliberate uses only, so a discard-as-cost never counts as "played".
        logs = observation.get("logs") or []
        bottoms = self._bottom_map(players) if self.capture_v23 else None
        # Absolute log indices for THIS seat's slice, resolved ONCE so the legacy loop below
        # and _observe_v23_logs agree on which entries are new (see the log_seen note in
        # __init__). Only computed under --heads v23; every other run keeps the old stream
        # byte-for-byte.
        fresh = None
        if self.capture_v23:
            seat = current["yourIndex"]
            base = self.log_seen[seat]
            self.log_seen[seat] = base + len(logs)
            fresh = [offset for offset in range(len(logs)) if base + offset >= self.log_high]
            if logs:
                self.log_high = max(self.log_high, base + len(logs))
        for offset, log in enumerate(logs):
            if log.get("type") in (10, 11, 12) and log.get("cardId") \
                    and log.get("playerIndex") is not None:
                self.events.append({"kind": "used", "player": log["playerIndex"],
                                    "move": move_index, "turn": turn,
                                    "card": log["cardId"]})
            elif log.get("type") == LOG_TYPE_ATTACK \
                    and log.get("playerIndex") is not None:
                if fresh is not None and offset not in fresh:
                    # The other seat's stream already reported this attack, at its own
                    # (later) move index and under its own turn. Emitting it twice made
                    # L1 count every attack as two exercised events and, worse, gave the
                    # copy a move index that can fall on the far side of an energy's span
                    # boundary (2026-08-06 audit: 168 logs -> 331 events, 1.97x).
                    continue
                attack_event = {"kind": "attack", "player": log["playerIndex"],
                                "move": move_index, "turn": turn,
                                "attack": log.get("attackId") or 0,
                                "card": log.get("cardId")}
                if self.capture_v23:
                    # WHICH Pokemon attacked, as the evolution-stable stack BOTTOM: the log
                    # names the stack TOP, so an attacker that evolved after being benched
                    # would otherwise look like a different Pokemon to the v23 labels.
                    serial = log.get("serial")
                    attack_event["serial"] = bottoms.get(serial, serial) \
                        if serial is not None else None
                self.events.append(attack_event)
        if self.capture_v23:
            self._observe_v23_logs(logs, observation, move_index, turn, bottoms, fresh)
            self.prev_bottoms = bottoms
        prizes = [len(players[0]["prize"]), len(players[1]["prize"])]
        boards = [_board_serials(players[0]), _board_serials(players[1])]
        tops = [{}, {}]
        serial_hp = [{}, {}]
        side_board_ids = [[], []]
        side_active_ids = [None, None]
        energy = {}
        attachments = {} if self.capture_v22 else None
        energy_serials = {}
        energy_total = [0, 0]
        for player_index in (0, 1):
            side_active_ids[player_index] = next(
                (pokemon["id"] for pokemon in (players[player_index].get("active") or [])
                 if pokemon is not None), None)
            for pokemon in ((players[player_index].get("active") or [])
                            + (players[player_index].get("bench") or [])):
                if pokemon is None:
                    continue
                stack = [pokemon["serial"]] + [
                    underneath["serial"]
                    for underneath in (pokemon.get("preEvolution") or [])
                    if isinstance(underneath, dict) and "serial" in underneath]
                tops[player_index][pokemon["serial"]] = (
                    pokemon["hp"], stack, pokemon.get("maxHp") or pokemon["hp"])
                attached = len(pokemon.get("energies") or [])
                side_board_ids[player_index].append(pokemon["id"])
                energy_total[player_index] += attached
                # v2.2: the attached CARD IDS (energy cards + tools), not a count. The v21
                # count above is the resolved EnergyType list and stays as it was: a single
                # Double Turbo Energy card is 1 id here and 2 energies there, so one cannot
                # be derived from the other.
                attached_ids = None
                if attachments is not None:
                    attached_ids = tuple(
                        card["id"] for card in (pokemon.get("energyCards") or [])) + tuple(
                        card["id"] for card in (pokemon.get("tools") or []))
                # Keyed by the stack's BASIC. `stack` is [top] + preEvolution, and
                # preEvolution is OLDEST-FIRST, so the Basic is stack[1] once the stack has
                # evolved at all -- stack[-1] was the STAGE 1 of a Stage-2 and the key
                # therefore changed at the second evolution (2026-08-06 audit; the same
                # defect as aux_head_labels.bottom_serial, and these two MUST agree or the
                # attach events and the label lookups key differently).
                energy_serials[stack[1] if len(stack) > 1 else stack[0]] = (
                    player_index, tuple(
                        (card["serial"], card["id"])
                        for card in (pokemon.get("energyCards") or [])))
                for serial in stack:
                    serial_hp[player_index][serial] = pokemon["hp"]
                    energy[serial] = attached          # evolution-safe: whole stack shares
                    if attachments is not None:
                        attachments[serial] = attached_ids
        if self.prev_prizes is not None:
            for player_index in (0, 1):
                taken = self.prev_prizes[player_index] - prizes[player_index]
                if taken > 0:
                    self.events.append({"kind": "prize", "player": player_index,
                                        "move": move_index, "turn": turn, "count": taken})
            for player_index in (0, 1):
                for serial in self.prev_board[player_index] - boards[player_index]:
                    self.events.append({"kind": "left", "player": player_index,
                                        "move": move_index, "turn": turn, "serial": serial,
                                        "hp": self.prev_serial_hp[player_index]
                                        .get(serial, 0)})
                if self.capture_v23:
                    # ...and the symmetric ARRIVAL, which the v23 bench labels key on: a
                    # PLAY option is only a bench drop if that hand card's serial actually
                    # entered play (mechanical -- no card-type lookup).
                    for serial in boards[player_index] - self.prev_board[player_index]:
                        self.events.append({"kind": "entered", "player": player_index,
                                            "move": move_index, "turn": turn,
                                            "serial": serial})
                for top_serial, (hp, stack, max_hp) in tops[player_index].items():
                    previous = self.prev_tops[player_index].get(top_serial)
                    # DAMAGE is a rise in (maxHp - hp), not a fall in hp. A stadium or tool
                    # that changes MAX hp moves `hp` with it -- losing Hero's Cape drops
                    # both by the same amount and is not a hit -- so the old `hp <
                    # previous.hp` test booked those swings as damage (2026-08-06 audit:
                    # 4-6 of ~665 damage events).
                    if previous is None or len(previous) < 3:
                        continue
                    taken = (max_hp - hp) - (previous[2] - previous[0])
                    if taken > 0:
                        self.events.append({"kind": "damage", "player": player_index,
                                            "move": move_index, "turn": turn,
                                            "amount": taken,
                                            "serials": stack})
        if self.prev_prizes is not None:
            # v23: per-energy identity diff (holder = stack bottom, evolution-stable).
            # Gained serial -> attach_seen with its holder; serial gone while the holder
            # is STILL in play -> energy_lost (retreat discard / effect cost / opponent
            # strip -- the label scan separates those by turn ownership). Holders that
            # left play take their energies with them and emit nothing (a KO is not
            # consumption).
            for bottom, (owner, cards) in energy_serials.items():
                previous = self.prev_energy_serials.get(bottom)
                if previous is None:
                    continue
                previous_cards = dict(previous[1])
                current_cards = dict(cards)
                for serial, card_id in cards:
                    if serial not in previous_cards:
                        self.events.append({"kind": "attach_seen", "player": owner,
                                            "move": move_index, "turn": turn,
                                            "holder": bottom, "serial": serial,
                                            "card": card_id})
                for serial, card_id in previous[1]:
                    if serial not in current_cards:
                        self.events.append({"kind": "energy_lost", "player": owner,
                                            "move": move_index, "turn": turn,
                                            "holder": bottom, "serial": serial,
                                            "card": card_id})
        self.prev_energy_serials = energy_serials
        for player_index in (0, 1):
            if self.deck_empty[player_index] is None \
                    and players[player_index].get("deckCount") == 0:
                self.deck_empty[player_index] = (move_index, turn)
                self.events.append({"kind": "deck_empty", "player": player_index,
                                    "move": move_index, "turn": turn})
        if self.prev_turn is not None and turn != self.prev_turn:
            # The state half of the snapshot is the LAST observation of the turn that just
            # ended (same convention as `hands`), not this new turn's first observation.
            snapshot = {"turn": self.prev_turn, "hands": list(self.last_hands),
                        "supporters": [_discard_supporters(players[0]),
                                       _discard_supporters(players[1])],
                        "board_ids": [list(self.last_board_ids[0]),
                                      list(self.last_board_ids[1])],
                        "active_ids": list(self.last_active_ids),
                        "energy": dict(self.last_energy),
                        "energy_total": list(self.last_energy_total),
                        "present": set(self.last_present)}
            if self.capture_v22:
                snapshot["attachments"] = dict(self.last_attachments)
                snapshot["stadium"] = self.last_stadium
            self.snapshots.append(snapshot)
        if self.capture_v22:
            # Turn TIMELINE: the first move index each turn was observed at. An action taken
            # during turn T is stamped with the POST-increment move counter, so the actions
            # of turn T occupy the move range (first_move[T], first_move[T + 1]].
            self.turn_first_move.setdefault(turn, move_index)
        if turn != self.prev_turn and turn not in self.turn_start_hands:
            self.turn_start_hands[turn] = [players[0].get("handCount") or 0,
                                           players[1].get("handCount") or 0]
        you = current["yourIndex"]
        mover_waiting = 1 - you                       # metas waiting on YOUR view
        if (self.pending[mover_waiting] and turn > self.pending_turn[mover_waiting]
                and players[you]["hand"] is not None):
            hand_ids = [card["id"] for card in players[you]["hand"]]
            active = next((p for p in (players[you].get("active") or [])
                           if p is not None), None)
            board_ids = [p["id"] for p in ((players[you].get("active") or [])
                                           + (players[you].get("bench") or []))
                         if p is not None]
            for meta in self.pending[mover_waiting]:
                meta["opp_hand_next_turn"] = hand_ids
                meta["opp_active_next_turn"] = active["id"] if active else None
                meta["opp_board_next_turn"] = board_ids
            self.pending[mover_waiting] = []
        if players[you]["hand"] is not None:
            hand = tuple(card["id"] for card in players[you]["hand"])
            if not self.hand_views[you] or self.hand_views[you][-1][1] != hand:
                self.hand_views[you].append((move_index, hand))
        self.prev_prizes = prizes
        self.prev_board = boards
        self.prev_tops = tops
        self.prev_serial_hp = serial_hp
        self.prev_turn = turn
        self.last_hands = [players[0].get("handCount") or 0,
                           players[1].get("handCount") or 0]
        self.last_board_ids = side_board_ids
        self.last_active_ids = side_active_ids
        self.last_energy = energy
        self.last_energy_total = energy_total
        self.last_present = boards[0] | boards[1]
        if self.capture_v22:
            self.last_attachments = attachments
            stadium = current.get("stadium") or []
            self.last_stadium = stadium[0]["id"] if stadium else 0

    def _bottom_map(self, players):
        """Every in-play serial (stack members included) -> its stack's BOTTOM serial, over
        BOTH sides. Carries the previous observation's map forward for stacks that have just
        left play, so a log naming a Pokemon KO'd by its own attack still resolves."""
        bottoms = dict(getattr(self, "prev_bottoms", None) or {})
        for player in players:
            for pokemon in ((player.get("active") or []) + (player.get("bench") or [])):
                if pokemon is None:
                    continue
                bottom = aux_head_labels.bottom_serial(pokemon)
                bottoms[pokemon["serial"]] = bottom
                for underneath in (pokemon.get("preEvolution") or []):
                    bottoms[underneath["serial"]] = bottom
        return bottoms

    def _observe_v23_logs(self, logs, observation, move_index, turn, bottoms=None,
                          fresh=None):
        """The v2.3 log-derived events, DEDUPED across the two seats' streams.

        `nextLogStart` hands seat p the global log slice [seen_p, len(logs)) and then advances
        seen_p, so counting what this seat has consumed gives every log its ABSOLUTE index;
        an index at or below the high-water mark has already been recorded from the other
        seat's stream and is dropped. Exact -- two identical events in a row are two indices.
        `fresh` is that offset list, resolved once by `observe` so the legacy attack-event
        loop there dedupes against exactly the same indices.

        Emitted: HP_CHANGE (with the engine's own putDamageCounter flag, the value, and the
        target's HP as of the previous board scan, which is what L2's ledger needs), cards
        ARRIVING in a hand (fetch tracking, by serial), and PLAY/EVOLVE by serial."""
        running = {}
        for offset in (fresh if fresh is not None else range(len(logs))):
            log = logs[offset]
            log_type = log.get("type")
            player = log.get("playerIndex")
            serial = log.get("serial")
            if log_type == LOG_TYPE_HP_CHANGE and serial is not None:
                # The ledger is keyed by the stack BOTTOM, like every other v23 stream: the
                # log names the stack TOP, while `decisive`'s row serial comes from
                # option_descriptors -> bottom_serial. Keying it by the top made every
                # placement onto an evolved Pokemon -- i.e. onto the opponent's actual
                # attackers -- unfindable, and `decisive` masked 41% of its rows, exactly
                # the evolved-target rate (2026-08-06 audit). prev_serial_hp holds EVERY
                # stack member, so the bottom resolves there too.
                key = (bottoms or {}).get(serial, serial)
                before = running.get(key)
                if before is None:
                    before = self.prev_serial_hp[0].get(key)
                    if before is None:
                        before = self.prev_serial_hp[1].get(key)
                value = log.get("value") or 0
                self.events.append({"kind": "hp", "player": player, "move": move_index,
                                    "turn": turn, "serial": key, "value": value,
                                    "counter": bool(log.get("putDamageCounter")),
                                    "before": before})
                if before is not None:
                    # HP_CHANGE.value is NEGATIVE for damage and positive for heals
                    # (measured: 98 negative / 12 positive over 15 games; -70 on a 100 HP
                    # Pokemon leaves 30). `before - value` ADDED the damage, so every
                    # second and later hit in one batch got a `before` that was wrong by
                    # 2x the first hit (2026-08-06 audit).
                    running[key] = before + value
            elif log_type in (LOG_TYPE_DRAW, LOG_TYPE_MOVE_CARD) and serial is not None \
                    and player is not None:
                if log_type == LOG_TYPE_DRAW or log.get("toArea") == AREA_HAND:
                    self.events.append({"kind": "hand_add", "player": player,
                                        "move": move_index, "turn": turn, "serial": serial})
                elif log.get("fromArea") == AREA_HAND:
                    # Left a hand WITHOUT being played (a play logs LOG_PLAY): opponent
                    # discard/shuffle disruption or our own discard cost. Consumed by the
                    # v25 fetch_delay label to mask never-played fetches whose rot evidence
                    # was destroyed. Additive: every event consumer dispatches by kind.
                    self.events.append({"kind": "hand_lost", "player": player,
                                        "move": move_index, "turn": turn, "serial": serial})
            elif log_type in (10, 12) and serial is not None:
                self.events.append({"kind": "used_serial", "player": player,
                                    "move": move_index, "turn": turn, "log": log_type,
                                    "serial": serial, "target": log.get("serialTarget")})

    def record_v23_state(self, mover, move_index, state):
        """--heads v23: this select's engine energy facts for the mover's own board (see
        aux_head_labels.decision_state). MAIN selects only -- attacks are declared there."""
        if state:
            self.v23_energy_state.append((move_index, mover, state))

    def record_knockouts(self, move_index, knockouts):
        """The build_v25 dump's CUMULATIVE KO record, diffed to `ko` events. Engine truth
        for what the prize join could only infer (and misinfer: a bounce within the join
        window of an unrelated prize take reads as a KO in every inference layer at once).

        The record names the stack's TOP card; the event carries the BOTTOM serial (the
        labels' key), resolved through the same carried-forward bottom map the HP ledger
        uses, so a stack KO'd by its own attack still resolves. `ko_scan_high` marks how
        far engine truth reaches -- label-side, departures beyond it fall back to the
        join (the terminal scan normally leaves no tail). None = no dump this scan (the
        v23/truth DLLs have no `knockouts` export): scan_high does NOT advance, so a
        whole game without the v25 engine labels exactly as before."""
        if knockouts is None:
            return
        self.ko_scan_high = move_index
        for record in list(knockouts)[self.ko_count:]:
            top = record.get("serial")
            self.events.append({
                "kind": "ko", "move": move_index, "turn": record.get("turn"),
                "serial": self.prev_bottoms.get(top, top), "top": top,
                "player": record.get("playerIndex"),
                "prizes": record.get("prizeCount", 0),
                "taker": record.get("takerIndex"),
                "byAttack": record.get("byAttack", 0)})
        self.ko_count = max(self.ko_count, len(knockouts))

    def record_selection(self, observation, select, indices, move_index):
        """The chosen options of ONE answered select: the exact ACTION stream of both seats
        (self-play drives both). `observation` is the PRE-select observation, so the acting
        player and the TURN are the ones the action was actually taken on.

        This is the only sound source for the action labels. The engine's `logs` are "since
        THIS player's last selection", so feeding both seats' observations into one stream
        reports every public action TWICE -- and the second copy arrives in an observation
        that has often already moved on to the next turn, which would teach the model that
        the opponent plays cards during our turn. The legacy log-derived "used" events are
        left exactly as they were (the legacy heads consume them as rest-of-game sets, where
        neither duplication nor the turn shift changes anything)."""
        current = observation["current"]
        mover = current["yourIndex"]
        turn = current["turn"]
        player = current["players"][mover]
        options = select.get("option") or []
        if select.get("context") == SELECT_CONTEXT_MAIN:
            if self.capture_main_offers:
                self._capture_main_offers(player, mover, options, move_index)
            if self.capture_v22:
                # Turn OWNERSHIP, straight off the select stream: the engine offers its MAIN
                # action menu to the player whose turn it is and to nobody else. This is the
                # marker the v2.2 windows are built on -- no turn-parity arithmetic anywhere.
                self.turn_owner.setdefault(turn, mover)
        for index in indices:
            if not 0 <= index < len(options):
                continue
            option = options[index]
            option_type = option.get("type")
            if option_type in (OPTION_TYPE_PLAY, OPTION_TYPE_ATTACH, OPTION_TYPE_EVOLVE):
                # deliberate uses only, exactly like the legacy PLAY/ATTACH/EVOLVE logs: a
                # card an EFFECT puts into play is a card move, not a play
                card_id = _option_card_id(player, option, default_area=AREA_HAND)
                if card_id:
                    self.events.append({"kind": "play", "player": mover,
                                        "move": move_index, "turn": turn, "card": card_id})
            elif option_type == OPTION_TYPE_ABILITY:
                card_id = _option_card_id(player, option)
                if card_id:
                    ability_event = {"kind": "ability", "player": mover,
                                     "move": move_index, "turn": turn, "card": card_id}
                    if self.capture_v23:
                        # WHICH Pokemon hosted it, as the evolution-stable bottom serial:
                        # the attach-energy CONDITION the v23 credit rule reads is a
                        # property of the host, not of the card id alone.
                        holder = {AREA_ACTIVE: player.get("active"),
                                  AREA_BENCH: player.get("bench")}.get(option.get("area"))
                        index = option.get("index")
                        if holder and index is not None and 0 <= index < len(holder) \
                                and holder[index] is not None:
                            ability_event["serial"] = aux_head_labels.bottom_serial(holder[index])
                    self.events.append(ability_event)
            elif option_type == OPTION_TYPE_ATTACK:
                self.events.append({"kind": "declare_attack", "player": mover,
                                    "move": move_index, "turn": turn,
                                    "attack": option.get("attackId") or 0})
            elif option_type == OPTION_TYPE_RETREAT:
                self.events.append({"kind": "retreat", "player": mover,
                                    "move": move_index, "turn": turn})

    def _capture_main_offers(self, player, mover, options, move_index):
        """ONE MAIN select's payability facts (--attack-aux-weight only): which attackIds
        and which HOSTS' abilities the engine offered, plus the mover's own board in the
        ENCODER's token emission order (its active, then its bench -- `encode_game`), so a
        label can be attached to the token row a head reads.

        An ABILITY option names its host by (area, index) and carries no skill identity
        (cg.api.OptionType.ABILITY), so the host is resolved to a SERIAL here -- the only
        key that survives a bench reordering between selects.

        Called for EVERY answered MAIN select, not just the ones the model decided: a
        forced/single-option MAIN select still reveals what was payable, and later
        decisions read those offers as their payable_next / payable_horizon labels.

        `move_index` is the caller's POST-increment move counter, while `_append_decision`
        stamps a decision's meta with the PRE-increment one -- so `move_index - 1` is the
        key the decisions taken at this select carry."""
        board_serials, board_card_ids = [], []
        host_serials = {}
        for area, slots in ((AREA_ACTIVE, player.get("active") or []),
                            (AREA_BENCH, player.get("bench") or [])):
            for index, pokemon in enumerate(slots):
                if pokemon is None:
                    continue
                host_serials[(area, index)] = pokemon["serial"]
                board_serials.append(pokemon["serial"])
                board_card_ids.append(pokemon["id"])
        abilities = {host_serials.get((option.get("area"), option.get("index")))
                     for option in options
                     if option.get("type") == OPTION_TYPE_ABILITY}
        abilities.discard(None)     # an ability hosted off my board (hand / discard): no
                                    # token row to hang a label on
        self.main_offers.append((
            move_index - 1, mover, _first_active_serial(player),
            tuple(board_serials), tuple(board_card_ids),
            tuple(option["attackId"] for option in options
                  if option.get("type") == OPTION_TYPE_ATTACK and option.get("attackId")),
            tuple(sorted(abilities))))

    def record_decision_state(self, mover, move_index, turn, prize_count, prize_ids,
                              locks, main_select):
        """ONE seat's engine-state facts at ONE of its own selects (--heads v22): how many
        prizes it has left, the ids in that pile when they are DEDUCIBLE, the 10 restriction
        bits, and whether this was a MAIN select. Recorded for BOTH seats -- including a
        frozen snapshot seat, whose decisions are never trained but whose prize pile is
        still the label the OTHER seat's `opp_prizes` head wants."""
        self.decision_facts[mover].append((move_index, turn, prize_count, prize_ids,
                                           locks, bool(main_select)))

    def register(self, mover, meta):
        self.pending[mover].append(meta)
        self.pending_turn[mover] = meta["turn"]


_cg_game = None
_cg_battle = None


class _Battle:
    """One live engine battle, addressed by ITS OWN pointer.

    `cg.game` keeps the current battle pointer in a module global, so a worker that
    interleaves several games (--games-per-worker) swaps its pointer in before every call.
    The engine's state AND its mt19937 live inside the battle object (engine_src/Game.h),
    so interleaved play is bit-identical to sequential play -- verified over 6 seeded games
    (every observation hashed, sequential vs all-6-alive round-robin: identical)."""

    __slots__ = ("pointer", "observation")

    def __init__(self, deck_0, deck_1, seed):
        global _cg_game, _cg_battle
        if _cg_game is None:
            from cg import game as cg_game
            from cg.sim import Battle as cg_battle
            _cg_game, _cg_battle = cg_game, cg_battle
        self.observation, _ = _cg_game.battle_start(deck_0, deck_1, seed=seed)
        self.pointer = _cg_battle.battle_ptr

    def select(self, move):
        _cg_battle.battle_ptr = self.pointer
        self.observation = _cg_game.battle_select(move)
        return self.observation

    def finish(self):
        _cg_battle.battle_ptr = self.pointer
        _cg_game.battle_finish()


def run_task(task):
    if task[0] == "selfplay":
        return play_selfplay_game(task)
    return play_probe_game(task)


def run_task_block(tasks):
    """A CHUNK of tasks played by one worker, up to --games-per-worker at a time. Selfplay
    tasks are pipelined (while one game's forward is in flight the worker steps the others);
    probe tasks stay one at a time. Returns one record per task, in COMPLETION order (the
    trainer aggregates records, it never indexes them by task)."""
    if any(task[0] != "selfplay" for task in tasks):
        return [run_task(task) for task in tasks]
    return _play_selfplay_block(tasks, int(_worker.get("concurrency") or 1))


# The reply a driver hands back to a game generator when its forward could not be served;
# the generator raises inside its own try/except, exactly as _policy_forward used to.
_FORWARD_FAILED = object()
_SERVER_TIMEOUT = 30.0
_SERVER_RETRY_TIMEOUT = 5.0
_SERVER_ATTEMPTS = 3               # 1 at 30 s + 2 at 5 s, matching _policy_forward


class _Forward:
    """One decision waiting on the net."""

    __slots__ = ("model", "tag", "encoded", "option_features", "payload", "attempts")

    def __init__(self, model, tag, encoded, option_features):
        self.model = model
        self.tag = tag
        self.encoded = encoded
        self.option_features = option_features
        self.payload = None
        self.attempts = 0

    def build_payload(self, index):
        return _server_payload(index, self.tag, self.encoded, self.option_features)

    def serve_local(self):
        if self.model is None:
            return _FORWARD_FAILED
        try:
            return _local_forward(self.model, self.encoded, self.option_features)
        except Exception:
            return _FORWARD_FAILED


class _LeafForward:
    """A SEARCH-LEAF BATCH waiting on the net: many rows, one queue message, one reply.

    Same duck type as `_Forward` for the driver (`_play_selfplay_block` does not care which
    it is), and the reply the generator receives is `(list of priors, list of values)`
    instead of `(priors, value)` -- which is exactly what `search_gen.run_search` expects
    back from a yield."""

    __slots__ = ("model", "tag", "rows", "payload", "attempts")

    def __init__(self, model, tag, rows):
        self.model = model
        self.tag = tag
        self.rows = rows              # [(tokens, owners, zones, globals, options, ids)]
        self.payload = None
        self.attempts = 0

    def build_payload(self, index):
        return (LEAF_TAG, index, self.tag[0], self.tag[1], self.rows)

    def serve_local(self):
        if self.model is None:
            return _FORWARD_FAILED
        try:
            priors, values = [], []
            for row in self.rows:
                encoded = {"token_features": row[0].astype(np.float32),
                           "owner_ids": row[1].astype(np.int64),
                           "zone_ids": row[2].astype(np.int64),
                           "global_features": row[3],
                           "card_ids": None if row[5] is None else row[5]}
                probabilities, value = _local_forward(self.model, encoded,
                                                      row[4].astype(np.float32))
                priors.append(probabilities)
                values.append(value)
            return priors, values
        except Exception:
            return _FORWARD_FAILED


def _submit_forward(forward):
    """Ship a forward to the GPU server. -> the request id, or None if it must be served
    locally (no server attached, no model tag, or the queue refused it)."""
    server = _worker.get("server")
    if server is None or forward.tag is None:
        return None
    request_queue, _reply_queue, index = server
    if forward.payload is None:
        forward.payload = forward.build_payload(index)
    try:
        sequence = _worker["sequence"] = _worker["sequence"] + 1
        request_queue.put(forward.payload + (sequence,))
        forward.attempts += 1
        return sequence
    except Exception:
        return None


def _serve_locally(forward):
    """The local-CPU fallback reply for one forward (or _FORWARD_FAILED)."""
    return forward.serve_local()


def _forward_failed(generator, forward, inflight, ready):
    """Server said no (error reply or timeout). Same policy as _policy_forward: retry the
    SAME payload with a fresh id when there is no local model, otherwise fall back to the
    worker's own CPU forward."""
    if _worker.get("no_model"):
        if forward.attempts < _SERVER_ATTEMPTS:
            sequence = _submit_forward(forward)
            if sequence is not None:
                inflight[sequence] = (generator, forward,
                                      time.monotonic() + _SERVER_RETRY_TIMEOUT)
                return
        ready.append((generator, _FORWARD_FAILED))
        return
    ready.append((generator, _serve_locally(forward)))


def _collect_replies(reply_queue, inflight, ready):
    """Block until at least one outstanding request is resolved, then drain whatever else
    has already arrived. Replies carry their request id (see _await_server_reply): one that
    belongs to an abandoned request is dropped instead of being applied to another game."""
    deadline = min(entry[2] for entry in inflight.values())
    replies = []
    try:
        replies.append(reply_queue.get(timeout=max(0.05, deadline - time.monotonic())))
        while True:
            replies.append(reply_queue.get_nowait())
    except Exception:
        pass
    for reply in replies:
        entry = inflight.pop(reply[0], None)
        if entry is None:
            continue                                   # reply to an abandoned request
        generator, forward, _deadline = entry
        if reply[1] is None:
            _forward_failed(generator, forward, inflight, ready)
        else:
            ready.append((generator, (reply[1], reply[2])))
    now = time.monotonic()
    for sequence in [key for key, entry in inflight.items() if entry[2] <= now]:
        generator, forward, _deadline = inflight.pop(sequence)
        _forward_failed(generator, forward, inflight, ready)


def _label_record(record):
    """Build this finished game's aux labels IN THE WORKER and ship them with the record
    (2026-07-31 pipeline rework). The main process then only concatenates; see
    `build_labels`. Label building is deterministic and side-effect free, so a failure here
    would be a bug, not a hiccup -- it is not caught."""
    if not (_worker.get("aux_labels") or _worker.get("payability")):
        return record
    stats = Counter()
    record["aux_entries"] = build_labels(
        record, heads=_worker.get("heads", "family1"),
        aux_labels=bool(_worker.get("aux_labels")),
        v21_modules=_worker.get("v21_modules") or ("token", "side", "action"),
        v21_checks=bool(_worker.get("v21_checks")),
        payability=bool(_worker.get("payability")), stats=stats)
    record["label_stats"] = dict(stats)
    return record


def _play_selfplay_block(tasks, concurrency):
    """Play `tasks` with up to `concurrency` games advancing at once. Each game is its own
    generator with its own rng, engine battle and trackers, so a game's trajectory depends
    only on its seed and the weights -- never on K or on how the games interleave."""
    if _worker.get("scripted") is not None:
        # The bundle opponent is one loaded module per worker and may keep per-game state;
        # interleaving games through it would corrupt that. Scripted mode stays serial.
        concurrency = 1
    server = _worker.get("server")
    reply_queue = server[1] if server is not None else None
    queued = list(tasks)
    ready = []                     # (generator, reply) pairs that can advance right now
    inflight = {}                  # request id -> (generator, forward, deadline)
    results = []
    # --search-gen: each live game owns a SEARCH SLOT, i.e. its own engine agent pointer,
    # so several trees can be open at once without two of them sharing one pointer's
    # session (the seed is per session -- interleaving inside a pointer breaks
    # determinism). The slot is recycled when the game finishes.
    free_slots = list(range(max(1, concurrency)))
    slot_of = {}
    while queued or ready or inflight:
        while ready or (queued and len(inflight) + len(ready) < concurrency):
            if ready:
                generator, reply = ready.pop()
            else:
                slot = free_slots.pop() if free_slots else 0
                generator = _selfplay_generator(queued.pop(0), slot)
                slot_of[id(generator)] = slot
                reply = None
            try:
                forward = generator.send(reply)
            except StopIteration as stop:
                slot = slot_of.pop(id(generator), None)
                if slot is not None:
                    free_slots.append(slot)
                results.append(_label_record(stop.value))
                continue
            sequence = _submit_forward(forward)
            if sequence is None:
                ready.append((generator, _serve_locally(forward)))
            else:
                inflight[sequence] = (generator, forward,
                                      time.monotonic() + _SERVER_TIMEOUT)
        if inflight:
            _collect_replies(reply_queue, inflight, ready)
    return results


def play_selfplay_game(task):
    """One self-play game. Returns the trainable decisions (numpy, f16 features) plus
    the outcome. When a snapshot is seated, its side's decisions are NOT returned."""
    return _play_selfplay_block([task], 1)[0]


def _decision_model(mover, snapshot_version, snapshot_side, version):
    """(model, tag, frozen) for one decision: which weights this seat plays with. Lifted
    out of _selfplay_generator so the v3 and v4 select paths resolve it identically."""
    frozen = snapshot_version is not None and mover == snapshot_side
    if frozen and _worker["no_model"]:
        # The server serves the ("snapshot", version) tag from the same file, so
        # loading a local copy per worker would only re-add what --no-worker-model
        # removed. A pruned file now surfaces as a server error -> counted `errors`.
        model = None
    elif frozen:
        try:
            model = _snapshot_model(snapshot_version)
        except Exception as error:
            # Snapshot file pruned while this pipelined block was in flight (or
            # stale dims): play the seat as the current model instead of dying.
            print(f"[worker] snapshot v{snapshot_version} unavailable "
                  f"({type(error).__name__}) -> current model", flush=True)
            frozen = False
    if not frozen:
        model = _worker["model"]
    return model, (("snapshot", snapshot_version) if frozen else ("current", version)), frozen


def _truth_engine():
    """This worker's DumpTrueState binding (--prize-labels truth). LABELS AND DIAGNOSTICS
    ONLY -- no code path from here reaches a determinization, a prior or a model input."""
    engine = _worker.get("truth_engine")
    if engine is None:
        from true_state import engine_for_cg
        engine = _worker["truth_engine"] = engine_for_cg()
        if "DumpTrueState" in engine.missing:
            raise RuntimeError("--prize-labels truth needs engine_src/build_truth/cg.dll")
    return engine


def _search_session(slot):
    """This worker's search session for one game slot (one engine agent pointer each),
    built on first use so a run without --search-gen never touches the search API."""
    if not _worker.get("search"):
        return None
    sessions = _worker.get("sessions")
    if sessions is None:
        import search_gen
        sessions = _worker["sessions"] = search_gen.session_factory(
            max(1, int(_worker.get("concurrency") or 1)))
    return sessions[slot % len(sessions)]


def _belief_deck(tracker, fallback_deck, cache):
    """The opponent's full deck, from the belief model (DeckRecognizer over the ladder
    corpus) constrained by everything this seat has actually seen -- the SAME machinery
    the inference wrapper determinizes with. Never DumpTrueState.

    Cached on the reveal multiset: the reveals only change when the opponent shows a new
    card, so a recognizer scan runs a handful of times per game instead of per decision."""
    revealed = tracker.revealed_counts()
    key = tuple(sorted(revealed.items()))
    hit = cache.get(key)
    if hit is not None:
        return hit
    deck = None
    try:
        recognizer = _worker.get("recognizer")
        if recognizer is None:
            from src.decks import DeckRecognizer
            recognizer = _worker["recognizer"] = DeckRecognizer()
        found = recognizer.consistent_deck(revealed)
        if found:
            deck = [int(card) for card, count in found.items() for _ in range(count)]
    except Exception:
        deck = None
    deck = deck or list(fallback_deck)
    cache[key] = deck
    return deck


def _search_decision(session, observation, select, priors, root_value, model, tag,
                     deck_counts, knowledge, history, in_flight, my_deck, opponent_deck,
                     settings, game_seed, decision_index):
    """One SEARCHED decision, as a sub-generator: yields `_LeafForward` batches to the
    driver and returns (SearchResult, SearchLeafEncoder) or None.

    The determinization is BELIEF-BASED and its rng is keyed to (game seed, turn, seat), so
    every searched decision inside one owned turn samples the SAME world -- the design's
    'one determinized world per own turn' -- while the engine session itself is opened and
    closed per decision (see the build report: carrying the subtree across real decisions
    was cut, the world is what is reused).
    """
    import search_gen
    from src.search.determinize import build_determinization
    current = observation["current"]
    mover = current["yourIndex"]
    world_rng = random.Random((int(game_seed) * 7919 + int(current["turn"]) * 31
                               + mover) & 0x7FFFFFFF)
    try:
        determinization = build_determinization(observation, my_deck, opponent_deck,
                                                world_rng, knowledge=knowledge)
    except Exception:
        return None
    seed = search_gen.derive_seed(game_seed, decision_index)
    simulations = settings["sims"]
    deep = random.Random(seed ^ 0x5BF03635).random() < settings["deep_p"]
    if deep:
        simulations = settings["sims_deep"]
    config = search_gen.SearchConfig(
        simulations=simulations, c_puct=settings["c_puct"], k_forced=settings["k_forced"],
        prior_cap=settings["prior_cap"], dirichlet_scale=settings["dirichlet_scale"],
        dirichlet_weight=settings["dirichlet_weight"],
        max_outstanding=settings["max_outstanding"], manual_coin=settings["manual_coin"],
        lazy_expand=settings["lazy_expand"])
    opened = session.begin(observation["search_begin_input"], determinization, seed,
                           manual_coin=config.manual_coin)
    if opened is None:
        return None
    root_id, root_options, root_mover = opened
    if root_options != len(select["option"]) or root_mover != mover:
        # The engine's own root menu must be the menu the agent is answering, or the visit
        # target would be indexed against a different option list. Refuse instead of
        # emitting a mislabelled target.
        session.end()
        return None
    encoder = SearchLeafEncoder(session, deck_counts, knowledge, history, in_flight)
    algorithm = search_gen.root_algorithm(
        settings["root_algo"], simulations, max_actions=settings["gumbel_actions"],
        sigma_scale=settings["gumbel_sigma"])
    tree = search_gen.run_search(
        session, root_id, mover, len(select["option"]), priors, encoder, config,
        random.Random(seed), noise=(settings["root_algo"] != "gumbel"),
        root_algo=algorithm, root_value=root_value)
    answer = None
    result = None
    try:
        while True:
            try:
                payloads = tree.send(answer)
            except StopIteration as stop:
                result = stop.value
                break
            reply = yield _LeafForward(model, tag, payloads)
            if reply is _FORWARD_FAILED:
                raise RuntimeError("leaf batch unavailable")
            answer = reply
    finally:
        tree.close()
        session.end()
    if result is not None:
        result.deep = deep
    return result, encoder


def _search_move(result, observation, rng, settings):
    """Which option a SEARCHED decision plays (addendum 10).

    PUCT: proportional to the PRUNED visits for the first --search-temp-turns turns of the
    game, argmax after -- trajectory diversity early, best-known play late. Gumbel: the
    action sequential halving ended on, which is the algorithm's own answer.
    Action-agnostic: it reads a distribution and an index."""
    if result.algorithm == "gumbel" and result.played is not None:
        return int(result.played)
    target = result.target
    if not target or sum(target) <= 0:
        return None
    if observation["current"]["turn"] <= settings["temp_turns"]:
        roll = rng.random()
        cumulative = 0.0
        for index, weight in enumerate(target):
            cumulative += weight
            if roll < cumulative:
                return index
        return len(target) - 1
    best = max(target)
    tied = [index for index, weight in enumerate(target) if weight >= best - 1e-12]
    return tied[rng.randrange(len(tied))] if len(tied) > 1 else tied[0]


def _append_decision(decisions, tracker, observation, mover, moves, encoded,
                     option_features, chosen, logprob, value, extra=None):
    """Record ONE model decision as a training sample -- the block that used to sit inline in
    _selfplay_generator, unchanged. v1/v2/v3 call it once per select; v4 calls it once per
    SUB-PICK, so every pick (and the STOP decision) gets its own logprob, its own chosen
    index over that step's candidate set, and its own value estimate."""
    current = observation["current"]
    board = []                    # mover-centric emission order = encoder's board
    for player_index in (mover, 1 - mover):
        player = current["players"][player_index]
        for pokemon in ((player.get("active") or []) + (player.get("bench") or [])):
            if pokemon is not None:
                board.append((pokemon["serial"], player_index, pokemon["hp"]))
    meta = {"move": moves, "turn": current["turn"],
            "active_serial": _first_active_serial(current["players"][mover]),
            "board": board, "opp_hand_next_turn": None,
            "opp_active_next_turn": None,
            "opp_board_next_turn": None}
    if extra:
        # --search-gen only: the searched flag, the pruned visit target and the search root
        # value ride in the decision's meta, so the decision TUPLE keeps its arity and
        # every existing consumer (assemble, the parity gate, old harnesses) is untouched.
        meta.update(extra)
    token_payload = encoded["token_features"].astype(np.float16)
    option_payload = option_features.astype(np.float16)
    if _worker.get("packed_transfer"):
        token_payload = pack_dense(token_payload)
        option_payload = pack_dense(option_payload)
    decisions.append((
        token_payload,
        encoded["owner_ids"].astype(np.int16),
        encoded["zone_ids"].astype(np.int16),
        encoded["global_features"].astype(np.float32),
        option_payload,
        mover, chosen, logprob, value, meta,
        None if encoded.get("card_ids") is None
        else encoded["card_ids"].astype(np.int16)))
    tracker.register(mover, meta)


def _selfplay_generator(task, slot=0):
    """The self-play loop as a coroutine: it yields a `_Forward` at every learnable
    decision and is resumed with `(probabilities, value)` (or `_FORWARD_FAILED`). Driving
    it synchronously reproduces the old play_selfplay_game exactly; driving several at once
    is what --games-per-worker / --trees-per-worker does.

    `slot` is this game's search slot (its own engine agent pointer); ignored without
    --search-gen."""
    _kind, seed, version, snapshot_version, snapshot_side = task
    _refresh_weights(version)
    rng = random.Random(seed)
    deck_rng = random.Random(seed * 31 + 7)
    decks = [_sample_our_deck(deck_rng), _sample_field_deck(deck_rng)]
    deck_counts = [dict(Counter(decks[0])), dict(Counter(decks[1]))]

    decisions = []
    errors = 0
    # v4 accounting (the no-auto-answer proof): how many prompts the model answered
    # (`forwards`) vs how many had exactly one legal answer (`forced_*`, `forced_pick`).
    v4_stats = Counter()
    capture_v22 = bool(_worker.get("v22"))
    capture_v23 = bool(_worker.get("v23"))
    tracker = EventTracker(capture_main_offers=bool(_worker.get("payability")),
                           capture_v22=capture_v22, capture_v23=capture_v23)
    deck_counters = [Counter(decks[0]), Counter(decks[1])]
    v23_state = [None]        # this select's (token facts, holder facts), see aux_head_labels

    def capture_decision_state(mover, rich_out):
        """--heads v22: this select's prize pile + restriction bits for the deciding seat,
        read off the dump_state decode `_encode_decision` just did (never a second one).

        --prize-labels truth reads the prize CONTENTS from DumpTrueState instead of the
        residual deduction. The deduction can only close when the seat's own deck happens
        to be enumerated, and its era propagation assumes prize contents cannot change
        while the count is unchanged -- which Redeemable Ticket (1114) and Team Rocket's
        Bother-Bot (1131) both violate. Truth is per decision, so neither limitation
        applies. LABELS ONLY: nothing on the decision path, the encoder or the
        determinization ever sees this."""
        rich_cards, rich_players = rich_out[0] if rich_out else (None, None)
        prizes = observation["current"]["players"][mover].get("prize") or []
        prize_ids = _v22_prize_ids(rich_cards, mover, deck_counters[mover], len(prizes))
        if _worker.get("prize_labels") == "truth":
            try:
                prize_ids = _truth_engine().prize_ids(battle.pointer, mover)
            except Exception:
                prize_ids = None
        main_select = (observation.get("select") or {}).get("context") \
            == SELECT_CONTEXT_MAIN
        tracker.record_decision_state(
            mover, moves, observation["current"]["turn"], len(prizes), prize_ids,
            _v22_lock_bits(rich_players, mover), main_select)
        if capture_v23:
            # The SAME decoded dump, read once more for the v2.3 facts the observation does
            # not expose: the resolved noAbility bit and the engine's per-attack effective
            # energy requirement (engine_src DumpState `attackCost` / `energyOrder`). No
            # second dump_state call, no engine call on the decision path.
            v23_state[0] = aux_head_labels.decision_state(rich_cards, observation, mover,
                                                     main_select)
            if v23_state[0] is not None and v23_state[0][1]:
                tracker.record_v23_state(mover, moves, v23_state[0][1])
            if len(rich_out or ()) > 1:
                tracker.record_knockouts(moves, rich_out[1].get("knockouts"))
    # v2 input trackers: ONE SET PER SEAT, each seeded with that seat's own decklist and fed
    # only that seat's observations. Sharing them (or seeding seat 1 with seat 0's list) would
    # hand the model information it cannot have on the ladder -- see scripts/audit_v2_leaks.py.
    seat_knowledge = seat_history = seat_encoder = seat_in_flight = None
    if _has_trackers():
        seat_knowledge = [CardKnowledge(Counter(decks[0])), CardKnowledge(Counter(decks[1]))]
        seat_history = [ActionHistory(extended=ENCODING in ("v5", "v6")),
                        ActionHistory(extended=ENCODING in ("v5", "v6"))]
        if ENCODING in ("v5", "v6"):
            seat_in_flight = [encode_inflight.InFlightTracker(), encode_inflight.InFlightTracker()]
        if _worker.get("encode_cache"):
            # per (game, seat): byte-identical to encode_observation_v2 / _v3, ~2.4x cheaper
            cached = CachedV3Encoder if ENCODING in ("v3", "v4") else CachedV2Encoder
            seat_encoder = [cached(), cached()]
    # --- search-in-the-loop generation (--search-gen). Everything below is inert when the
    # flag is off: `search_settings` is None and not one branch is taken. ---
    search_settings = _worker.get("search")
    search_enabled = bool(search_settings) and ENCODING in ("v5", "v6") \
        and snapshot_version is None and _worker.get("scripted") is None
    search_session = _search_session(slot) if search_enabled else None
    search_enabled = search_enabled and search_session is not None
    search_rng = random.Random(seed * 6151 + 17)     # its OWN stream: the raw-policy
    search_stats = Counter()                         # sampling path is bit-unchanged
    decided_streak = [0, 0]                          # per seat: |V| > threshold in a row
    belief_trackers = belief_cache = None
    if search_enabled:
        from src.decks import OpponentTracker
        belief_trackers = [OpponentTracker(), OpponentTracker()]
        belief_cache = [{}, {}]
    battle = _Battle(list(decks[0]), list(decks[1]), seed)
    observation = battle.observation
    tracker.observe(observation, 0)
    moves = 0
    scripted = _worker.get("scripted")
    # --- Resignation (--resign-threshold, 0 = off). Both seats' critics must AGREE the
    # game is decided: the hopeless seat has scored <= -T on K consecutive of ITS OWN
    # decisions while the other has scored >= +T on K consecutive of ITS OWN. One-sided
    # despair is not enough -- that is exactly what a mis-calibrated critic looks like.
    # DISABLED for the whole game when a snapshot opponent or a scripted seat is seated:
    # the trigger reads two critics of the SAME current model, and a frozen/scripted
    # seat's numbers (or absence of numbers) are not comparable evidence.
    # A --resign-audit-fraction slice of triggers plays on to natural completion instead,
    # so the comeback rate of the trigger is measured continuously rather than assumed.
    resign_threshold = _worker.get("resign_threshold") or 0.0
    resign_enabled = (resign_threshold > 0.0 and snapshot_version is None
                      and scripted is None)
    resign_persist = _worker.get("resign_persist") or 6
    audit_rng = random.Random(seed * 7919 + 13)   # own stream: never perturbs play
    losing_streak = [0, 0]           # per seat: consecutive own decisions <= -T
    winning_streak = [0, 0]          # per seat: consecutive own decisions >= +T
    resigned = False
    audit = False
    trigger_move = None
    hopeless_seat = None
    forced_outcome = None
    while observation["current"]["result"] == -1 and moves < MOVE_CAP:
        seat = observation["current"]["yourIndex"]
        if _has_trackers() and not (scripted and seat == 1):
            seat_knowledge[seat].update(observation)
            seat_history[seat].update(observation)
        if belief_trackers is not None:
            # One tracker per seat, fed only that seat's own observations: the belief that
            # determinizes seat s's search may only contain what seat s has been shown.
            belief_trackers[seat].update(observation)
        select = observation.get("select")
        if select is None:
            if seat_in_flight is not None:        # no select = no chain in progress
                seat_in_flight[0].reset()
                seat_in_flight[1].reset()
            observation = battle.select([])
            moves += 1
            tracker.observe(observation, moves)
            continue
        if scripted and seat == 1:
            # Scripted seat answers EVERY select itself -- including take-all/multi-pick
            # ones _trivial_move would grab -- because e.g. which cards to discard is a
            # real expert decision, not a formality. Its decisions are never trained on.
            try:
                move = scripted(observation)
            except Exception:
                errors += 1
                move = _random_legal(select, rng)
            answered = observation
            try:
                observation = battle.select(move)
            except Exception:
                move = _random_legal(select, rng)
                observation = battle.select(move)
            moves += 1
            tracker.record_selection(answered, select, move, moves)
            tracker.observe(observation, moves)
            continue
        move = None
        if seat_in_flight is not None:
            seat_in_flight[seat].observe(observation, select)
        if ENCODING in ("v4", "v5", "v6"):
            # --- v4 SEQUENTIAL SELECTION (owner directive 2026-07-29): the model answers
            # every prompt that has more than one legal answer, one pick at a time, and
            # decides itself when to STOP. Same loop as encode_selection.resolve_with (v5:
            # encode_inflight.resolve_with_v5) -- written inline here because the forward must
            # be YIELDED to the pipelining driver. Keep the two in step.
            forced, reason = encode_selection.forced_answer(select)
            if forced is not None:
                move = forced                       # exactly ONE legal answer exists
                v4_stats["forced"] += 1
                v4_stats["forced_" + reason] += 1
            else:
                mover = observation["current"]["yourIndex"]
                model, tag, frozen = _decision_model(mover, snapshot_version,
                                                     snapshot_side, version)
                state = encode_selection.MultiSelect(observation, select)
                rich_out = [] if capture_v22 else None
                try:
                    # The board cannot change inside the loop (no battle_select until the
                    # answer is complete), so state + v3 option rows are encoded ONCE.
                    encoded, base_options = _encode_decision(
                        observation, deck_counts[mover], select["option"],
                        knowledge=None if seat_knowledge is None else seat_knowledge[mover],
                        history=None if seat_history is None else seat_history[mover],
                        encoder=None if seat_encoder is None else seat_encoder[mover],
                        in_flight=None if seat_in_flight is None else seat_in_flight[mover],
                        rich_out=rich_out)
                    if capture_v22:
                        capture_decision_state(mover, rich_out)
                    # What each ENGINE option physically refers to (serials), so a v23 label
                    # can be hung on the candidate ROW the scorer read. Built once per select.
                    descriptors = aux_head_labels.option_descriptors(observation, select) \
                        if capture_v23 else None
                    base_globals = encoded["global_features"]
                    option_extras = None
                    if ENCODING in ("v5", "v6"):
                        mover_in_flight = (None if seat_in_flight is None
                                           else seat_in_flight[mover])
                        option_extras = np.stack([
                            encode_inflight.option_extra_v5(observation, option, mover_in_flight)
                            for option in select["option"]]).astype(np.float32)
                        if ENCODING == "v6":
                            # ...++ the two chain-progress columns, i.e. exactly what
                            # state_encoder.base_option_matrix_v6 appends (split apart here
                            # because the v3 rows were encoded above).
                            # ALWAYS ask the ENGINE what each damage-placement option
                            # actually does (fork + step + read the resulting HP). This is
                            # not optional: without it V6_PROJECTED_HP falls back to the
                            # target's CURRENT hp, which the board tokens already carry, so
                            # the column would silently contain nothing -- no error, nothing
                            # in the logs. Self-play knows BOTH decks, so the determinization
                            # is the truth and never fails (measured 162/162). Costs ~0.4 s
                            # per 512-game iteration. Returns None on a non-single-pick
                            # select or an engine error; the column then reports the present
                            # state, never an invented one.
                            engine_hp = engine_projected_hp(
                                observation, select, decks[mover], decks[1 - mover],
                                rng=deck_rng)
                            option_extras = np.concatenate(
                                [option_extras,
                                 state_encoder.chain_columns(observation, select,
                                                         mover_in_flight, engine_hp)],
                                axis=1)
                    # Opt-in action rules (owner 2026-08-09): mirrors resolve_with_v6's
                    # counter mask -- the loops must stay in step. None unless enabled.
                    counter_mask = (state_encoder.combined_option_mask(observation, select)
                                    if ENCODING == "v6" else None)
                    while not state.complete():
                        forced_index = state.forced_index()
                        if forced_index is not None:
                            v4_stats["forced_pick"] += 1     # one option left, STOP illegal
                            state.take(forced_index)
                            continue
                        pending = state.pending()
                        if counter_mask is not None:
                            masked_pending = [i for i in pending if i in counter_mask]
                            if masked_pending:
                                v4_stats["counter_masked"] += (len(pending)
                                                               - len(masked_pending))
                                pending = masked_pending
                        stop_offered = state.stop_offered()
                        if ENCODING == "v6":
                            rows = state_encoder.candidate_matrix_v6(
                                base_options, option_extras, state, pending, stop_offered)
                        elif ENCODING == "v5":
                            rows = encode_inflight.candidate_matrix_v5(
                                base_options, option_extras, state, pending, stop_offered)
                        else:
                            rows = encode_selection.candidate_matrix(base_options, state, pending,
                                                              stop_offered)
                        step = dict(encoded, global_features=encode_selection.global_features(
                            base_globals, state))
                        reply = yield _Forward(model, tag, step, rows)
                        if reply is _FORWARD_FAILED:
                            raise RuntimeError("forward unavailable")
                        probabilities, value = reply
                        v4_stats["forwards"] += 1
                        # --- SEARCH-IN-THE-LOOP ------------------------------------- #
                        # A decision is searched iff (a) --search-gen is on, (b) it is a
                        # single-pick select -- the branch point the tree is built on --
                        # (c) an INDEPENDENT coin says so (--search-p; a random gate, never
                        # a confidence heuristic), and (d) the position is not already
                        # decided. The root's priors and value are the forward that just
                        # returned, so search costs no extra root evaluation.
                        chosen = None
                        extra = None
                        # `not stop_offered` = minCount >= 1: the candidate rows are then
                        # exactly the engine's options, so a visit target over the tree's
                        # edges indexes the recorded rows one-for-one.
                        searchable = (search_enabled and state.max_take == 1
                                      and state.count >= 2 and not state.chosen
                                      and not stop_offered)
                        if searchable:
                            search_stats["searchable"] += 1
                            if abs(value) > search_settings["decided_threshold"]:
                                decided_streak[mover] += 1
                            else:
                                decided_streak[mover] = 0
                        if searchable and search_rng.random() < search_settings["p"] \
                                and decided_streak[mover] < search_settings[
                                    "decided_persist"]:
                            opponent_deck = _belief_deck(
                                belief_trackers[mover], decks[mover],
                                belief_cache[mover])
                            outcome = yield from _search_decision(
                                search_session, observation, select, list(probabilities),
                                value, model, tag, deck_counts[mover],
                                seat_knowledge[mover], seat_history[mover],
                                None if seat_in_flight is None else seat_in_flight[mover],
                                decks[mover], opponent_deck, search_settings, seed, moves)
                            if outcome is None:
                                search_stats["rejected"] += 1
                            else:
                                result, leaf_encoder = outcome
                                search_stats["searched"] += 1
                                search_stats["sims"] += result.simulations
                                search_stats["leaves"] += result.leaves
                                search_stats["prior_requests"] += result.prior_requests
                                search_stats["cache_hits"] += result.cache_hits
                                search_stats["batches"] += result.batches
                                search_stats["encodes"] += leaf_encoder.encodes
                                search_stats["option_encodes"] += \
                                    leaf_encoder.option_encodes
                                search_stats["deep"] += int(bool(result.deep))
                                chosen = _search_move(result, observation, search_rng,
                                                      search_settings)
                                target = list(result.target or [])
                                # KL(visit target || the policy that generated it), used
                                # ONLY as the replay sampling weight: how much this
                                # position's teacher disagrees with the student that
                                # produced it, measured at insertion time.
                                divergence = 0.0
                                for index, weight in enumerate(target):
                                    if weight > 0 and index < len(probabilities):
                                        divergence += weight * float(np.log(
                                            weight / max(float(probabilities[index]),
                                                         1e-10)))
                                extra = {"searched": True, "search_target": target,
                                         "search_value": float(result.root_value),
                                         "search_deep": bool(result.deep),
                                         "search_kl": max(0.0, divergence)}
                                search_stats["kl_sum"] += max(0.0, divergence)
                        if chosen is None:
                            chosen = _sample_index(probabilities, rng)
                        if not frozen:
                            # Every SUB-PICK is its own training sample: own candidate set,
                            # own chosen index, own logprob, own value.
                            if descriptors is not None:
                                # ...and its own ROW->option map: the candidate rows ARE
                                # `pending` in order, plus the STOP row (which refers to no
                                # option, so it describes nothing and stays masked).
                                rows_v23 = tuple(descriptors[index] for index in pending
                                                 if index < len(descriptors))
                                if stop_offered:
                                    rows_v23 = rows_v23 + ((aux_head_labels.ROW_OTHER, 0, 0, 0),)
                                extra = dict(extra or {}, v23_rows=rows_v23,
                                             v23_context=select.get("context"),
                                             v23_state=(v23_state[0] or (None, None))[0])
                            _append_decision(
                                decisions, tracker, observation, mover, moves, step, rows,
                                chosen,
                                float(np.log(max(probabilities[chosen], 1e-10))), value,
                                extra=extra)
                        # Resignation reads ONE judgement per select (the FIRST sub-pick), so
                        # "K consecutive own decisions" keeps exactly its v3 meaning.
                        if resign_enabled and trigger_move is None and not state.chosen:
                            losing_streak[mover] = losing_streak[mover] + 1 \
                                if value <= -resign_threshold else 0
                            winning_streak[mover] = winning_streak[mover] + 1 \
                                if value >= resign_threshold else 0
                            for seat in (0, 1):
                                if losing_streak[seat] < resign_persist \
                                        or winning_streak[1 - seat] < resign_persist:
                                    continue
                                trigger_move, hopeless_seat = moves, seat
                                audit = audit_rng.random() < \
                                    _worker.get("resign_audit_fraction", 0.0)
                                if not audit:
                                    resigned = True
                                    forced_outcome = 1.0 if hopeless_seat == 1 else -1.0
                                break
                            if resigned:
                                break        # this decision is kept; the move is not played
                        if stop_offered and chosen == len(pending):
                            v4_stats["stop"] += 1
                            break            # the MODEL ended the answer
                        state.take(pending[chosen])
                    if resigned:
                        break
                    move = state.answer()          # in PICK order: it is the answer for an
                    v4_stats["selects"] += 1       # ordering prompt (SKILL_ORDER, to-deck),
                    v4_stats["picks"] += len(move)  # and the engine accepts non-ascending lists
                except Exception:
                    # Same trade as the v3 path (TRAINING audit MINOR-1): sub-picks already
                    # appended keep their labels while the trajectory follows a random answer,
                    # so the divergence is COUNTED here instead of hiding in the data.
                    errors += 1
                    move = _random_legal(select, rng)
        else:
            move = _trivial_move(select)
        if move is None:
            mover = observation["current"]["yourIndex"]
            model, tag, frozen = _decision_model(mover, snapshot_version, snapshot_side,
                                                 version)
            rich_out = [] if capture_v22 else None
            try:
                encoded, option_features = _encode_decision(
                    observation, deck_counts[mover], select["option"],
                    knowledge=None if seat_knowledge is None else seat_knowledge[mover],
                    history=None if seat_history is None else seat_history[mover],
                    encoder=None if seat_encoder is None else seat_encoder[mover],
                    rich_out=rich_out)
                if capture_v22:
                    capture_decision_state(mover, rich_out)
                # hand the request to the driver and suspend: with --games-per-worker > 1
                # the worker steps its OTHER games while this one waits on the server.
                reply = yield _Forward(model, tag, encoded, option_features)
                if reply is _FORWARD_FAILED:
                    raise RuntimeError("forward unavailable")
                probabilities, value = reply
                chosen = _sample_index(probabilities, rng)
                if not frozen:
                    _append_decision(decisions, tracker, observation, mover, moves,
                                     encoded, option_features, chosen,
                                     float(np.log(max(probabilities[chosen], 1e-10))),
                                     value)
                move = [chosen]
                if resign_enabled and trigger_move is None:   # fires at most once
                    # Streaks are per SEAT and count that seat's OWN decisions only, so
                    # "K consecutive" means K of its own turns' worth of judgement.
                    losing_streak[mover] = losing_streak[mover] + 1 \
                        if value <= -resign_threshold else 0
                    winning_streak[mover] = winning_streak[mover] + 1 \
                        if value >= resign_threshold else 0
                    for seat in (0, 1):
                        if losing_streak[seat] < resign_persist \
                                or winning_streak[1 - seat] < resign_persist:
                            continue
                        trigger_move, hopeless_seat = moves, seat
                        audit = audit_rng.random() < \
                            _worker.get("resign_audit_fraction", 0.0)
                        if not audit:
                            resigned = True
                            # Score it exactly as the engine would have: the OTHER seat
                            # won. result == 0 means seat 0 won, so outcome (seat-0
                            # centric) is +1 when seat 1 is the one giving up.
                            forced_outcome = 1.0 if hopeless_seat == 1 else -1.0
                        break
                    if resigned:
                        break            # this decision is kept; the move is not played
            except Exception:
                errors += 1
                move = _random_legal(select, rng)
        answered = observation
        try:
            observation = battle.select(move)
        except Exception:
            # The engine rejected the move we RECORDED. The trajectory now follows a random
            # one while `decisions` still holds `chosen` + its logprob, so the policy label
            # and the played action disagree for that row (TRAINING audit MINOR-1). Unlike
            # every other fallback in this file it used to be silent -- count it, so a
            # poisoning path shows up in the iter line's `errors` instead of hiding.
            errors += 1
            move = _random_legal(select, rng)
            observation = battle.select(move)
        if seat_in_flight is not None:
            # The engine ACCEPTED `move` -- record it as the in-flight source (fact-safe:
            # accepted picks only). `answered` is the observation the select belonged to.
            try:
                seat_in_flight[answered["current"]["yourIndex"]].record(answered, select,
                                                                        move)
            except Exception:
                errors += 1
        moves += 1
        tracker.record_selection(answered, select, move, moves)
        tracker.observe(observation, moves)
    result = observation["current"]["result"]
    if capture_v23:
        # Terminal knockout scan: the game-winning KO resolves AFTER the last decision's
        # dump, so without one extra dump here it would carry no engine record and fall
        # back to the prize join -- a mixed-source seam on exactly the decisive KO.
        try:
            terminal_extras = {}
            rich_state(observation, extras_out=terminal_extras)
            tracker.record_knockouts(moves, terminal_extras.get("knockouts"))
        except Exception:
            pass
    battle.finish()
    outcome = 1.0 if result == 0 else (-1.0 if result == 1 else 0.0)
    comeback = None
    if resigned:
        outcome = forced_outcome
    elif audit:
        # The trigger said this seat was lost; the game was played out anyway. A win OR a
        # draw for it is a comeback -- the safety gauge for the threshold.
        comeback = outcome >= 0.0 if hopeless_seat == 0 else outcome <= 0.0
    return {"kind": "selfplay", "seed": seed, "outcome": outcome, "moves": moves,
            "first_player": observation["current"].get("firstPlayer", -1),
            "decisions": decisions, "events": tracker.events,
            "snapshots": tracker.snapshots,
            "main_offers": tracker.main_offers,
            "turn_owner": tracker.turn_owner,
            "turn_first_move": tracker.turn_first_move,
            "decision_facts": tracker.decision_facts,
            "v23_energy_state": tracker.v23_energy_state,
            "v23_ko_scan_high": tracker.ko_scan_high,
            "turn_start_hands": tracker.turn_start_hands,
            "hand_views": tracker.hand_views,
            "deck_ids": [sorted(set(decks[0])), sorted(set(decks[1]))],
            "snapshot_side": snapshot_side if snapshot_version is not None else None,
            "errors": errors, "v4": dict(v4_stats), "search": dict(search_stats),
            "prize_labels": _worker.get("prize_labels", "deduction"),
            "resign": {"resigned": resigned, "audit": audit,
                       "trigger_move": trigger_move, "hopeless_seat": hopeless_seat,
                       "comeback": comeback}}


@torch.no_grad()
def _v4_probe_move(observation, select, deck_counts, knowledge, history, version, rng,
                   in_flight=None, engine_decks=None):
    """The v4/v5 selection loop for the GREEDY probe / eval: the SHARED loop
    (encode_selection.resolve_with / encode_inflight.resolve_with_v5), argmax with uniform tie-break
    instead of sampling (ties broken by engine order would measure a different policy from
    the one generation trains -- DECISION_SURFACE C5), and nothing recorded."""
    tag = ("current", version) if version is not None else None
    forced, _reason = encode_selection.forced_answer(select)
    if forced is not None:
        return forced
    if ENCODING in ("v5", "v6"):
        encoded = encode_inflight.encode_observation_v5(observation, deck_counts=deck_counts,
                                                  knowledge=knowledge, history=history,
                                                  in_flight=in_flight)
    else:
        encoded = encode_observation_v3(observation, deck_counts=deck_counts,
                                        knowledge=knowledge, history=history)
    base_globals = encoded["global_features"]

    def choose(rows, state):
        step = dict(encoded,
                    global_features=encode_selection.global_features(base_globals, state))
        probabilities, _value = _serve_forward(_worker["model"], tag, step, rows)
        return argmax_tiebreak(probabilities, rng)

    if ENCODING == "v6":
        # The probe must see the SAME option surface generation trains on, or it measures
        # the policy on inputs it never learned from.
        engine_hp = None
        if engine_decks:
            engine_hp = engine_projected_hp(observation, select, engine_decks[0],
                                            engine_decks[1], rng=rng)
        return state_encoder.resolve_with_v6(observation, select, choose, in_flight,
                                         engine_hp=engine_hp)
    if ENCODING == "v5":
        return encode_inflight.resolve_with_v5(observation, select, choose, in_flight)
    return encode_selection.resolve_with(observation, select, choose)


ENERGY_TYPE_DARKNESS = 7            # cg.api.EnergyType.DARKNESS


def _max_attack_cost(card_id):
    """How many energies the most expensive of a Pokemon's own attacks asks for. Pure
    card-table lookup, used ONLY to count surplus attachments in finished games."""
    from src.cards import get_attack, get_card
    card = get_card(int(card_id)) or {}
    costs = [len((get_attack(attack_id) or {}).get("energies") or [])
             for attack_id in (card.get("attacks") or [])]
    return max(costs) if costs else 0


class BehaviorCounters:
    """EVAL INSTRUMENTATION, never a decision path (design doc, 'Behavioral counters').

    Counts, from ONE finished probe game and one seat's observed play: how often it used an
    ability, how many energies it attached beyond what the holder's own attacks can spend,
    and where its Darkness energy ended up. Everything is derived by DIFFING successive
    observations -- no rule, no heuristic and no hook in `select` -- so switching it on
    cannot change a single move.
    """

    __slots__ = ("seat", "abilities", "attachments", "surplus", "dark_targets",
                 "_attached")

    def __init__(self, seat=0):
        self.seat = seat
        self.abilities = 0
        self.attachments = 0
        self.surplus = 0
        self.dark_targets = Counter()
        self._attached = {}            # serial -> (card id, attached energy card ids)

    def record_selection(self, observation, select, indices):
        if observation["current"]["yourIndex"] != self.seat:
            return
        options = select.get("option") or []
        for index in indices:
            if 0 <= index < len(options) \
                    and options[index].get("type") == OPTION_TYPE_ABILITY:
                self.abilities += 1

    def observe(self, observation):
        from src.cards import get_card
        player = observation["current"]["players"][self.seat]
        for pokemon in ((player.get("active") or []) + (player.get("bench") or [])):
            if pokemon is None:
                continue
            serial = pokemon["serial"]
            attached = [card["id"] for card in (pokemon.get("energyCards") or [])]
            previous = self._attached.get(serial)
            if previous is not None:
                gained = Counter(attached) - Counter(previous[1])
                for card_id, count in gained.items():
                    self.attachments += count
                    if (get_card(int(card_id)) or {}).get("energyType") \
                            == ENERGY_TYPE_DARKNESS:
                        self.dark_targets[pokemon["id"]] += count
                # "Surplus" = energies now on the card that its OWN attacks cannot spend.
                surplus = len(pokemon.get("energies") or []) \
                    - _max_attack_cost(pokemon["id"])
                previous_surplus = max(0, previous[2] if len(previous) > 2 else 0)
                if surplus > previous_surplus:
                    self.surplus += surplus - previous_surplus
                self._attached[serial] = (pokemon["id"], attached, max(0, surplus))
            else:
                self._attached[serial] = (
                    pokemon["id"], attached,
                    max(0, len(pokemon.get("energies") or [])
                        - _max_attack_cost(pokemon["id"])))

    def summary(self):
        return {"abilities": self.abilities, "attachments": self.attachments,
                "surplus_attachments": self.surplus,
                "dark_targets": dict(self.dark_targets)}


def play_probe_game(task):
    """Current policy (GREEDY) as player 0 vs a panel opponent. Eval only, never trained."""
    _kind, seed, version, opponent_name = task
    from cg import game
    from eval_panel import load_opponent
    # --no-worker-model probes run entirely on the GPU server, which needs a version to
    # name the weight file. Without one there is no model anywhere -- say so up front.
    if _worker["model"] is None and version is None:
        raise RuntimeError("probe has no model: --no-worker-model needs a versioned probe "
                           "so the forwards can go to the GPU server")
    if version is not None:
        _refresh_weights(version)
    if opponent_name not in _worker["opponents"]:
        if opponent_name == "specialist":      # the loaded bundle agent
            _worker["opponents"]["specialist"] = (_worker["bundle_agent"],
                                                  list(_worker["bundle_deck"]))
        else:
            _worker["opponents"][opponent_name] = load_opponent(opponent_name)
    opponent_agent, opponent_deck = _worker["opponents"][opponent_name]
    rng = random.Random(seed)
    deck_rng = random.Random(seed * 31 + 7)
    our_deck = _sample_our_deck(deck_rng)
    if opponent_deck is None:
        opponent_deck = _sample_field_deck(deck_rng)
    our_counts = dict(Counter(our_deck))
    knowledge = history = in_flight = None
    if _has_trackers():                # match training inputs: trackers live from move 0
        knowledge = CardKnowledge(Counter(our_deck))
        history = ActionHistory(extended=ENCODING in ("v5", "v6"))
        if ENCODING in ("v5", "v6"):
            in_flight = encode_inflight.InFlightTracker()

    observation, _ = game.battle_start(list(our_deck), list(opponent_deck), seed=seed)
    behavior = BehaviorCounters(seat=0)
    behavior.observe(observation)
    moves = 0
    while observation["current"]["result"] == -1 and moves < MOVE_CAP:
        if knowledge is not None and observation["current"]["yourIndex"] == 0:
            knowledge.update(observation)
            history.update(observation)
        select = observation.get("select")
        if select is None:
            if in_flight is not None:
                in_flight.reset()
            observation = game.battle_select([])
            moves += 1
            continue
        if observation["current"]["yourIndex"] == 0 and ENCODING in ("v4", "v5", "v6"):
            if in_flight is not None:
                in_flight.observe(observation, select)
            try:
                move = _v4_probe_move(observation, select, our_counts, knowledge, history,
                                      version, rng, in_flight=in_flight,
                                      engine_decks=(our_deck, opponent_deck))
            except Exception:
                move = _random_legal(select, rng)
        elif observation["current"]["yourIndex"] == 0:
            move = _trivial_move(select)
            if move is None:
                try:
                    probabilities, _value, _encoded, _options = _policy_forward(
                        _worker["model"], observation, our_counts, select["option"],
                        solver=_worker["solver"],
                        tag=("current", version) if version is not None else None,
                        knowledge=knowledge, history=history)
                    # Greedy, but EXACT ties broken uniformly instead of by engine option
                    # order -- otherwise the probe measures a different policy from the one
                    # generation trains (DECISION_SURFACE C5 / src/tiebreak.py).
                    move = [argmax_tiebreak(probabilities, rng)]
                except Exception:
                    move = _random_legal(select, rng)
        else:
            try:
                move = opponent_agent(observation)
            except Exception:
                move = _random_legal(select, rng)
        answered = observation
        try:
            observation = game.battle_select(move)
        except Exception:
            move = _random_legal(select, rng)
            observation = game.battle_select(move)
        if in_flight is not None and answered["current"]["yourIndex"] == 0:
            try:
                in_flight.record(answered, select, move)
            except Exception:
                pass                   # probe-only marker; never let it kill the game
        behavior.record_selection(answered, select, move)
        behavior.observe(observation)
        moves += 1
    result = observation["current"]["result"]
    game.battle_finish()
    return {"kind": "probe", "opponent": opponent_name,
            "win": 1.0 if result == 0 else 0.0,
            "behavior": behavior.summary()}


# ----------------------------------------------------------------------------------- #
# Learner side.
# ----------------------------------------------------------------------------------- #

def gae(values, terminal_reward, gamma, lam):
    """Sparse-terminal GAE over one player's own-decision trajectory."""
    length = len(values)
    advantages = np.zeros(length, dtype=np.float32)
    running = 0.0
    for t in reversed(range(length)):
        if t == length - 1:
            delta = terminal_reward - values[t]
        else:
            delta = gamma * values[t + 1] - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    return advantages


def _turn_bucket(delta):
    return min((max(delta, 0) + 1) // 2, 7)


def _clock_bucket(meta, events, player):
    """Own-side turns until PLAYER's next prize after this decision (grounded derivation)."""
    future_turns = [event["turn"] for event in events
                    if event["kind"] == "prize" and event["player"] == player
                    and event["move"] > meta["move"]]
    if not future_turns:
        return NUM_TIMING_CLASSES - 1                                  # never
    return _turn_bucket(min(future_turns) - meta["turn"])


def _milestone_buckets(meta, events, player, index=None):
    """Turns until PLAYER takes its k-th NEXT prize, k = 1..MILESTONES."""
    buckets = [NUM_TIMING_CLASSES - 1] * MILESTONES
    if index is not None:
        # Same events, off the per-PRIZE-CARD columns _v21_index expands once per game (a
        # take of 2 is two entries): the prizes after this decision are the suffix from one
        # bisect, and only the first MILESTONES of them can land in a bucket.
        moves = index["prize_move_column"][player]
        turns = index["prize_turn_column"][player]
        start = bisect.bisect_right(moves, meta["move"])
        for offset, turn in enumerate(turns[start:start + MILESTONES]):
            buckets[offset] = _turn_bucket(turn - meta["turn"])
        return buckets
    cumulative = 0
    for event in sorted((e for e in events if e["kind"] == "prize"
                         and e["player"] == player and e["move"] > meta["move"]),
                        key=lambda e: e["move"]):
        for _ in range(event.get("count", 1)):
            if cumulative < MILESTONES:
                buckets[cumulative] = _turn_bucket(event["turn"] - meta["turn"])
            cumulative += 1
    return buckets


def _is_ko(events, left_event):
    return any(other["kind"] == "prize" and other["player"] == 1 - left_event["player"]
               and abs(other["move"] - left_event["move"]) <= KO_JOIN_MOVES
               for other in events)


def _ko_clock_labels(meta, events, index=None):
    """Per board token (meta['board'] order = the encoder's board-token order): turns until
    that Pokemon's stack is KO'd; 'never' if it survives / merely bounces. -1 pads."""
    labels = np.full(MAX_BOARD_TOKENS, -1, dtype=np.int64)
    if index is not None:
        # The first departure of THIS serial after the decision, off the per-serial list
        # (move-ordered, KO flag already resolved) instead of a scan of the whole stream.
        left = index["left"]
        move, turn = meta["move"], meta["turn"]
        for position, (serial, owner, _hp) in enumerate(meta.get("board") or ()):
            if position >= MAX_BOARD_TOKENS:
                break
            labels[position] = NUM_TIMING_CLASSES - 1                  # never
            for event_move, event_turn, _event_hp, knocked_out in left.get((owner, serial),
                                                                          ()):
                if event_move > move:
                    if knocked_out:
                        labels[position] = _turn_bucket(event_turn - turn)
                    break
        return labels
    left_by_serial = {}
    for event in events:
        if event["kind"] == "left" and event["move"] > meta["move"]:
            left_by_serial.setdefault((event["player"], event["serial"]), event)
    for position, (serial, owner, _hp) in enumerate(meta.get("board") or []):
        if position >= MAX_BOARD_TOKENS:
            break
        event = left_by_serial.get((owner, serial))
        if event is not None and _is_ko(events, event):
            labels[position] = _turn_bucket(event["turn"] - meta["turn"])
        else:
            labels[position] = NUM_TIMING_CLASSES - 1                  # never
    return labels


def _damage_labels(meta, events, index=None):
    """Per board token: overflow-capped damage RECEIVED within the next 1/2/3 own rounds
    (2/4/6 global turns). Chip damage comes from public HP drops (evolution-line safe: a
    stack's damage credits every serial in it, and evolving's HP gain is not damage); a KO
    adds exactly the remaining HP (overflow beyond the KO is lost, per the design doc).
    Cap = HP at decision time. -1 pads; /DAMAGE_CAP."""
    labels = np.full((MAX_BOARD_TOKENS, DAMAGE_WINDOWS), -1.0, dtype=np.float32)
    if index is not None:
        # Per serial the damage events are move-ordered AND turn-ordered (the whole stream
        # is -- _v21_index refuses to build otherwise), so "after this move, up to this
        # turn" is a contiguous slice and its total is a difference of two prefix sums.
        # Integer sums throughout: identical to the Python sum() below, bit for bit.
        move, turn = meta["move"], meta["turn"]
        for position, (serial, owner, hp_now) in enumerate(meta.get("board") or ()):
            if position >= MAX_BOARD_TOKENS:
                break
            knockout_turn = knockout_hp = None
            for event_move, event_turn, event_hp, knocked_out in \
                    index["left"].get((owner, serial), ()):
                if event_move > move and knocked_out:
                    knockout_turn, knockout_hp = event_turn, event_hp
                    break
            series = index["damage"].get((owner, serial))
            start = 0 if series is None else bisect.bisect_right(series[0], move)
            for window in range(DAMAGE_WINDOWS):
                turn_limit = turn + 2 * (window + 1)
                total = 0
                if series is not None:
                    end = bisect.bisect_right(series[1], turn_limit)
                    if end > start:
                        total = series[2][end] - series[2][start]
                if knockout_turn is not None and knockout_turn <= turn_limit:
                    total += knockout_hp
                labels[position, window] = min(float(total), float(hp_now)) / DAMAGE_CAP
        return labels
    for position, (serial, owner, hp_now) in enumerate(meta.get("board") or []):
        if position >= MAX_BOARD_TOKENS:
            break
        ko_event = None
        for event in events:
            if (event["kind"] == "left" and event["player"] == owner
                    and event["serial"] == serial and event["move"] > meta["move"]
                    and _is_ko(events, event)):
                ko_event = event
                break
        for window in range(DAMAGE_WINDOWS):
            turn_limit = meta["turn"] + 2 * (window + 1)
            total = sum(event["amount"] for event in events
                        if event["kind"] == "damage" and event["player"] == owner
                        and serial in event["serials"] and event["move"] > meta["move"]
                        and event["turn"] <= turn_limit)
            if ko_event is not None and ko_event["turn"] <= turn_limit:
                total += ko_event.get("hp", 0)
            labels[position, window] = min(float(total), float(hp_now)) / DAMAGE_CAP
    return labels


def _deckout_buckets(meta, events, mover, index=None):
    buckets = []
    for player in (mover, 1 - mover):
        if index is not None:
            empty = next(((move, turn) for move, turn in index["deck_empty"][player]
                          if move > meta["move"]), None)
            buckets.append(NUM_TIMING_CLASSES - 1 if empty is None
                           else _turn_bucket(empty[1] - meta["turn"]))
            continue
        empty = next((e for e in events if e["kind"] == "deck_empty"
                      and e["player"] == player and e["move"] > meta["move"]), None)
        buckets.append(NUM_TIMING_CLASSES - 1 if empty is None
                       else _turn_bucket(empty["turn"] - meta["turn"]))
    return buckets


def _hand_size_labels(meta, snapshots, mover):
    """[my hand at end of my turn, mine at end of opp's next turn, theirs at end of their
    next turn], /HAND_SIZE_CAP; -1 = masked (game ended first)."""
    by_turn = {s["turn"]: s for s in snapshots}
    labels = np.full(3, -1.0, dtype=np.float32)
    now = by_turn.get(meta["turn"])
    then = by_turn.get(meta["turn"] + 1)
    if now is not None:
        labels[0] = min(now["hands"][mover], HAND_SIZE_CAP) / HAND_SIZE_CAP
    if then is not None:
        labels[1] = min(then["hands"][mover], HAND_SIZE_CAP) / HAND_SIZE_CAP
        labels[2] = min(then["hands"][1 - mover], HAND_SIZE_CAP) / HAND_SIZE_CAP
    return labels


def _supporter_label(meta, snapshots, mover, by_turn=None):
    """Card id of the supporter the OPPONENT plays on their next turn (their discard grows
    by it), 0 = none; -1 = masked (game ended before their turn resolved)."""
    if by_turn is None:
        by_turn = {s["turn"]: s for s in snapshots}
    now = by_turn.get(meta["turn"])
    then = by_turn.get(meta["turn"] + 1)
    if now is None or then is None:
        return -1
    before = now["supporters"][1 - mover]
    after = then["supporters"][1 - mover]
    for card_id, count in after.items():
        if count > before.get(card_id, 0):
            return int(card_id)
    return 0


def _survives_label(meta, events, mover, index=None):
    if meta["active_serial"] is None:
        return -1.0                                                    # masked
    if index is not None:
        for event_move, event_turn, _event_hp, knocked_out in \
                index["left"].get((mover, meta["active_serial"]), ()):
            if event_move > meta["move"] and event_turn <= meta["turn"] + 1 \
                    and knocked_out:
                return 0.0
        return 1.0
    for event in events:
        if (event["kind"] == "left" and event["player"] == mover
                and event["serial"] == meta["active_serial"]
                and event["move"] > meta["move"]
                and event["turn"] <= meta["turn"] + 1):
            if _is_ko(events, event):
                return 0.0
    return 1.0


def _hand_multihot(card_ids):
    if card_ids is None:
        return None
    hot = np.zeros(CARD_VOCAB, dtype=np.float32)
    for card_id in card_ids:
        if 0 < card_id < CARD_VOCAB:
            hot[card_id] = 1.0
    return hot


def _hand_flags(card_ids):
    from src.cards import get_card
    if card_ids is None:
        return np.full(HAND_FLAGS, -1.0, dtype=np.float32)             # masked
    flags = np.zeros(HAND_FLAGS, dtype=np.float32)
    for card_id in card_ids:
        card = get_card(card_id) or {}
        card_type = card.get("cardType")
        if card_type == 0 and card.get("basic"):
            flags[0] = 1.0
        if card.get("stage1") or card.get("stage2"):
            flags[1] = 1.0
        if card_type == 1:
            flags[2] = 1.0
        if card_type == 3:
            flags[3] = 1.0
        if card_type in (5, 6):
            flags[4] = 1.0
    return flags


# --- v2.1 labels (NEXT_MODEL_DESIGN.md section 3) ------------------------------------
# Horizon h => turn meta["turn"] + h, and the side that acts on it: h 0 = the mover, still
# inside its own turn ("rest of this turn"); h 1 = the opponent's next turn; h 2 = the
# mover's next turn. Everything is hindsight-exact and may use privileged information --
# only the model's INPUTS are restricted.

def _empty_turn_actions():
    return {"play_moves": [], "play_cards": [], "ability_moves": [], "ability_cards": [],
            "declare_moves": [], "declare_attacks": [], "retreat_moves": []}


def _v21_index(events):
    """Per-GAME index of the event stream, built once and shared by that game's ~160
    decisions. Every v2.1 label function filters the stream on the same few keys (player,
    serial, turn, "after this move"), so each key gets its own move-ordered list here and a
    decision answers by bisecting it instead of walking ~1500 events. Nothing is
    approximated: the lists hold exactly the events the scans would have visited.

    Returns None -- and every label function then falls back to its original scan -- if the
    stream is not move- AND turn-ordered. The tracker appends in move order and the engine's
    turn never goes backwards, so this is a tripwire, not an expected path."""
    previous_move = previous_turn = -1
    for event in events:
        if event["move"] < previous_move or event["turn"] < previous_turn:
            return None
        previous_move, previous_turn = event["move"], event["turn"]
    prize_moves, prize_turns, prize_counts = ([], []), ([], []), ([], [])
    deck_empty = ([], [])
    play_moves, play_cards = ([], []), ([], [])
    # Per-player GLOBAL action columns (move-ordered), the v2.2 windows' only input: a window
    # is a move RANGE, so every window on every decision is two bisects and a slice.
    ability_moves, ability_cards = ([], []), ([], [])
    attack_moves, attack_ids = ([], []), ([], [])
    retreat_moves = ([], [])
    left, damage, turn_actions, attack_log = {}, {}, {}, {}
    for event in events:
        kind = event["kind"]
        player = event["player"]
        move, turn = event["move"], event["turn"]
        if kind == "damage":
            for serial in event["serials"]:
                series = damage.get((player, serial))
                if series is None:
                    series = damage[(player, serial)] = ([], [], [0])
                series[0].append(move)
                series[1].append(turn)
                series[2].append(series[2][-1] + event["amount"])       # prefix sums
        elif kind == "left":
            left.setdefault((player, event["serial"]), []).append(
                [move, turn, event.get("hp", 0), False])
        elif kind == "prize":
            prize_moves[player].append(move)
            prize_turns[player].append(turn)
            prize_counts[player].append(event.get("count", 1))
        elif kind == "play":
            play_moves[player].append(move)                 # rest-of-game use integrals
            play_cards[player].append(event["card"])
            if event["card"]:                               # per-turn action window
                actions = turn_actions.get((turn, player))
                if actions is None:
                    actions = turn_actions[(turn, player)] = _empty_turn_actions()
                actions["play_moves"].append(move)
                actions["play_cards"].append(event["card"])
        elif kind == "ability":
            if event["card"]:
                ability_moves[player].append(move)
                ability_cards[player].append(event["card"])
                actions = turn_actions.get((turn, player))
                if actions is None:
                    actions = turn_actions[(turn, player)] = _empty_turn_actions()
                actions["ability_moves"].append(move)
                actions["ability_cards"].append(event["card"])
        elif kind == "declare_attack":
            attack_moves[player].append(move)
            attack_ids[player].append(event["attack"])
            actions = turn_actions.get((turn, player))
            if actions is None:
                actions = turn_actions[(turn, player)] = _empty_turn_actions()
            actions["declare_moves"].append(move)
            actions["declare_attacks"].append(event["attack"])
        elif kind == "retreat":
            retreat_moves[player].append(move)
            actions = turn_actions.get((turn, player))
            if actions is None:
                actions = turn_actions[(turn, player)] = _empty_turn_actions()
            actions["retreat_moves"].append(move)
        elif kind == "deck_empty":
            deck_empty[player].append((move, turn))
        elif kind == "attack":
            attack_log.setdefault((turn, player), []).append(event["attack"])
    # A departure is a KO iff the OTHER side takes a prize within KO_JOIN_MOVES moves of it
    # -- decision-independent, so it is resolved once here instead of per board token.
    for (player, _serial), entries in left.items():
        opponent_prizes = prize_moves[1 - player]
        for entry in entries:
            entry[3] = (bisect.bisect_right(opponent_prizes, entry[0] + KO_JOIN_MOVES)
                        > bisect.bisect_left(opponent_prizes, entry[0] - KO_JOIN_MOVES))
    # Whole-turn action labels (horizons 1/2 have no move filter, so they are per-turn
    # constants); horizon 0 slices the *_moves lists above.
    for actions in turn_actions.values():
        actions["plays"] = sorted(set(actions["play_cards"]))
        actions["abilities"] = sorted(set(actions["ability_cards"]))
        actions["attack"] = next((value for value in actions["declare_attacks"]
                                  if 0 < value < ATTACK_VOCAB), 0)
        actions["retreat"] = float(bool(actions["retreat_moves"]))
        # ...and the SUFFIX form horizon 0 reads. Built backwards once per turn, so a
        # decision does one bisect and reuses a shared list instead of running its own
        # sorted(set(...)) over the slice (speed pass 2: same values, same objects reused).
        actions["play_suffix"] = _suffix_sets(actions["play_cards"])
        actions["ability_suffix"] = _suffix_sets(actions["ability_cards"])
        actions["attack_suffix"] = _suffix_first_attack(actions["declare_attacks"])
    # Prize takes EXPANDED to one entry per prize card (an event can take 2-3 at once), so
    # "turns until my k-th next prize" is a bisect plus a slice of MILESTONES entries.
    prize_turn_column = ([], [])
    prize_move_column = ([], [])
    for player in (0, 1):
        for position, move in enumerate(prize_moves[player]):
            for _ in range(prize_counts[player][position]):
                prize_move_column[player].append(move)
                prize_turn_column[player].append(prize_turns[player][position])
    # Rest-of-game distinct plays per side: the suffix sets only change at the ~60 moves
    # that introduce a new card, so those are the only sorted() calls, and the resulting
    # list is shared by every suffix that has the same set.
    use_suffix = ([], [])
    for player in (0, 1):
        cards = play_cards[player]
        suffix = [None] * (len(cards) + 1)
        current, seen = [], set()
        suffix[len(cards)] = current
        for position in range(len(cards) - 1, -1, -1):
            if cards[position] not in seen:
                seen.add(cards[position])
                current = sorted(seen)
            suffix[position] = current
        use_suffix[player].extend(suffix)
    return {"left": left, "damage": damage,
            "prize_moves": prize_moves,
            "prize_turns": prize_turns, "prize_counts": prize_counts,
            "prize_move_column": prize_move_column,
            "prize_turn_column": prize_turn_column,
            "deck_empty": deck_empty, "turn_actions": turn_actions,
            "attack_log": attack_log, "play_moves": play_moves,
            "play_cards": play_cards,
            "ability_moves": ability_moves, "ability_cards": ability_cards,
            "attack_moves": attack_moves, "attack_ids": attack_ids,
            "retreat_moves": retreat_moves,
            "use_suffix": use_suffix}


def _suffix_sets(cards):
    """cards[k:] -> its sorted distinct set, for every k, as ONE shared list per distinct
    suffix (the `use_suffix` construction, reused per turn). suffix[len(cards)] is empty."""
    suffix = [None] * (len(cards) + 1)
    current, seen = [], set()
    suffix[len(cards)] = current
    for position in range(len(cards) - 1, -1, -1):
        if cards[position] not in seen:
            seen.add(cards[position])
            current = sorted(seen)
        suffix[position] = current
    return suffix


def _suffix_first_attack(attacks):
    """attacks[k:] -> its first in-vocabulary attack id (0 = none), for every k."""
    suffix = [0] * (len(attacks) + 1)
    for position in range(len(attacks) - 1, -1, -1):
        suffix[position] = (attacks[position]
                            if 0 < attacks[position] < ATTACK_VOCAB
                            else suffix[position + 1])
    return suffix


def _v21_context(record):
    """The per-GAME lookups every decision of that game shares (built once per record)."""
    by_turn = {}
    for event in record.get("events") or []:
        by_turn.setdefault(event["turn"], []).append(event)
    start_hands = {int(turn): hands for turn, hands
                   in (record.get("turn_start_hands") or {}).items()}
    hand_views = record.get("hand_views") or [[], []]
    # Each seat's hand views are appended in move order, so the "latest view at or before
    # this decision" scan becomes a bisect over these move columns.
    view_moves = [[move for move, _hand in views] for views in hand_views]
    if any(list(moves) != sorted(moves) for moves in view_moves):
        view_moves = None                                   # tripwire: keep the scan
    return {"by_turn": by_turn,
            # Which seat OWNS a turn. Horizons are turn offsets, so the side they describe
            # follows turn ownership, not `mover` parity: a seat can make a trainable
            # decision on the OTHER seat's turn (forced select after a KO, effect response),
            # measured at 0.42% of decisions, and those rows got exactly inverted horizon
            # labels (TRAINING audit MAJOR-1). first_player < 0 (never observed) falls back
            # to the old parity rule.
            "first_player": record.get("first_player", -1),
            "snapshots": {snapshot["turn"]: snapshot
                          for snapshot in (record.get("snapshots") or [])},
            "snapshot_list": record.get("snapshots") or [],
            "start_hands": start_hands,
            "hand_views": hand_views,
            "hand_view_moves": view_moves,
            "deck_ids": record.get("deck_ids") or [[], []],
            "turns_started": set(start_hands) | set(by_turn),
            "index": _v21_index(record.get("events") or [])}


def _horizon_side(context, turn, mover, horizon):
    """The seat whose turn `turn` is -- i.e. the side a horizon's action/hand labels
    describe. Turn parity <-> seat is exact in this engine (0 violations over 459 turns);
    `mover` is NOT, because forced selects happen on the opponent's turn."""
    first_player = context.get("first_player", -1)
    if first_player is not None and first_player >= 0 and turn > 0:
        return first_player if turn % 2 == 1 else 1 - first_player
    return mover if horizon % 2 == 0 else 1 - mover


def _v21_token_labels(meta, events, context):
    """Per board token (meta['board'] order): KO turns + damage windows (the existing
    grounded derivations), attached energy at +1/+2, and still-in-play at +2."""
    board = (meta.get("board") or [])[:MAX_BOARD_TOKENS]
    energy = np.full((MAX_BOARD_TOKENS, 2), -1.0, dtype=np.float32)
    present = np.full(MAX_BOARD_TOKENS, -1.0, dtype=np.float32)
    for horizon in (1, 2):
        snapshot = context["snapshots"].get(meta["turn"] + horizon)
        if snapshot is None:                        # game ended first -> masked
            continue
        attached, in_play = snapshot["energy"], snapshot["present"]
        for index, (serial, _owner, _hp) in enumerate(board):
            energy[index, horizon - 1] = min(attached.get(serial, 0),
                                             ENERGY_CAP) / ENERGY_CAP
            if horizon == 2:
                present[index] = 1.0 if serial in in_play else 0.0
    return {"v21_ko": _ko_clock_labels(meta, events, context["index"]),
            "v21_damage": _damage_labels(meta, events, context["index"]),
            "v21_energy": energy,
            "v21_present": present}


def _v21_side_labels(meta, events, context, mover):
    """Per side: prize/deckout clocks, board composition and energy totals at +1/+2, the
    two hand sizes the spec keeps, and the opponent's hidden hand / deck / next active."""
    board_ids, slot = [], 0
    board_mask = np.zeros(V21_BOARD_SLOTS, dtype=bool)
    side_energy = np.full(V21_BOARD_SLOTS, -1.0, dtype=np.float32)
    for player in (mover, 1 - mover):               # slots: mine +1, mine +2, theirs +1/+2
        for horizon in (1, 2):
            snapshot = context["snapshots"].get(meta["turn"] + horizon)
            board_ids.append([] if snapshot is None
                             else list(snapshot["board_ids"][player]))
            if snapshot is not None:
                board_mask[slot] = True
                side_energy[slot] = min(snapshot["energy_total"][player],
                                        SIDE_ENERGY_CAP) / SIDE_ENERGY_CAP
            slot += 1
    hand_sizes = np.full(2, -1.0, dtype=np.float32)     # [ours at +2 start, theirs at +1]
    ours = context["start_hands"].get(meta["turn"] + 2)
    theirs = context["start_hands"].get(meta["turn"] + 1)
    # Same turn-ownership rule as the action horizons (MAJOR-1): +2 is the current turn
    # owner's next turn, +1 is the other seat's -- which is `mover` only when the mover
    # owns this turn.
    owner = _horizon_side(context, meta["turn"], mover, 0)
    if ours is not None:
        hand_sizes[0] = min(ours[owner], HAND_SIZE_CAP) / HAND_SIZE_CAP
    if theirs is not None:
        hand_sizes[1] = min(theirs[1 - owner], HAND_SIZE_CAP) / HAND_SIZE_CAP
    # The opponent's CURRENT hidden hand: their own latest observation at or before this
    # decision (a player's hand is fully visible on its own observations).
    opponent_hand = None
    views = context["hand_views"][1 - mover]
    if context["hand_view_moves"] is not None:
        position = bisect.bisect_right(context["hand_view_moves"][1 - mover],
                                       meta["move"]) - 1
        if position >= 0:
            opponent_hand = views[position][1]
    else:
        for move, hand in views:
            if move > meta["move"]:
                break
            opponent_hand = hand
    # Opponent's next active: read off the +1 snapshot (the end of THEIR next turn), the
    # same horizon convention as everything else here. Deliberately NOT the legacy
    # meta["opp_active_next_turn"] capture, whose shared pending watermark can hand a meta
    # a view one turn further out than its own +1 (measured on 8 of 143,380 decisions).
    next_turn = context["snapshots"].get(meta["turn"] + 1)
    active_id = None if next_turn is None else next_turn["active_ids"][1 - mover]
    return {"v21_prize": np.array(_milestone_buckets(meta, events, mover, context["index"])
                                  + _milestone_buckets(meta, events, 1 - mover,
                                                       context["index"]),
                                  dtype=np.int64),
            "v21_deckout": np.array(_deckout_buckets(meta, events, mover, context["index"]),
                                    dtype=np.int64),
            "v21_board_ids": board_ids, "v21_board_mask": board_mask,
            "v21_side_energy": side_energy, "v21_hand_sizes": hand_sizes,
            "v21_opp_hand_ids": None if opponent_hand is None else list(opponent_hand),
            "v21_opp_deck_ids": context["deck_ids"][1 - mover],
            "v21_opp_active": int(active_id) if active_id
            and 0 < active_id < CARD_VOCAB else -1}


def _v21_action_labels(meta, events, context, mover):
    """Per horizon (rest-of-this-turn / +1 / +2), for the side that acts on it: every card
    played, every ability used, the attack declared (class 0 = none), and whether it
    retreated; plus the retained rest-of-game per-card use integrals.

    Everything here comes from the SELECTION stream (`EventTracker.record_selection`), the
    only source with exact turn and actor attribution -- see the note there on why the
    engine's log stream is not usable for per-turn action labels."""
    index = context["index"]
    plays, abilities = [], []
    attack = np.full(V21_HORIZONS, -1, dtype=np.int64)
    retreat = np.full(V21_HORIZONS, -1.0, dtype=np.float32)
    mask = np.zeros(V21_HORIZONS, dtype=bool)
    for horizon in range(V21_HORIZONS):
        turn = meta["turn"] + horizon
        side = _horizon_side(context, turn, mover, horizon)
        if turn not in context["turns_started"]:    # game ended first -> masked horizon
            plays.append([])
            abilities.append([])
            continue
        mask[horizon] = True
        if index is not None:
            # Horizons 1/2 span a whole turn, so their sets are per-(turn, side) constants
            # (built once per game); horizon 0 is the same lists sliced at this move.
            actions = index["turn_actions"].get((turn, side))
            if actions is None:
                plays.append([])
                abilities.append([])
                attack[horizon] = 0
                retreat[horizon] = 0.0
                continue
            if horizon:
                plays.append(list(actions["plays"]))
                abilities.append(list(actions["abilities"]))
                attack[horizon] = actions["attack"]
                retreat[horizon] = actions["retreat"]
                continue
            # Horizon 0 = "the rest of THIS turn": one bisect per column into the suffix
            # forms _v21_index built for this turn (same values the slice-and-sort produced).
            plays.append(actions["play_suffix"][
                bisect.bisect_right(actions["play_moves"], meta["move"])])
            abilities.append(actions["ability_suffix"][
                bisect.bisect_right(actions["ability_moves"], meta["move"])])
            attack[horizon] = actions["attack_suffix"][
                bisect.bisect_right(actions["declare_moves"], meta["move"])]
            retreat[horizon] = float(bisect.bisect_right(actions["retreat_moves"],
                                                         meta["move"])
                                     < len(actions["retreat_moves"]))
            continue
        window = [event for event in context["by_turn"].get(turn, ())
                  if event["player"] == side
                  and (horizon > 0 or event["move"] > meta["move"])]
        plays.append(sorted({event["card"] for event in window
                             if event["kind"] == "play" and event["card"]}))
        abilities.append(sorted({event["card"] for event in window
                                 if event["kind"] == "ability" and event["card"]}))
        declared = [event["attack"] for event in window
                    if event["kind"] == "declare_attack"
                    and 0 < event["attack"] < ATTACK_VOCAB]
        attack[horizon] = declared[0] if declared else 0    # explicit "none" class
        retreat[horizon] = float(any(event["kind"] == "retreat" for event in window))
    if index is not None:
        use_ids = [list(index["use_suffix"][player][
            bisect.bisect_right(index["play_moves"][player], meta["move"])])
            for player in (mover, 1 - mover)]
    else:
        use_ids = [sorted({event["card"] for event in events
                           if event["kind"] == "play" and event["player"] == player
                           and event["move"] > meta["move"]})
                   for player in (mover, 1 - mover)]
    return {"v21_play_ids": plays, "v21_ability_ids": abilities,
            "v21_attack": attack, "v21_retreat": retreat, "v21_action_mask": mask,
            "v21_my_use_ids": use_ids[0], "v21_opp_use_ids": use_ids[1],
            # Does horizon 1 describe the OPPONENT? Almost always yes, but not on a forced
            # off-turn select (the mover's own next turn lands at +1) and not during setup
            # (turn 0, where +1 is the first player's turn, which may be the mover's). The
            # seat-swap tripwire in v21_batch_checks assumes "+1 == 1 - mover", so it must
            # skip those rows rather than fail them (TRAINING audit MAJOR-1).
            # Not a label: no head reads it, and the verifier compares its own key set.
            "v21_h1_opponent": bool(
                _horizon_side(context, meta["turn"] + 1, mover, 1) == 1 - mover)}


# --- v2.2 labels (SEARCH_TRAINING_DESIGN.md, "Aux head rework (v22 suite)") -----------
# MOVE-INDEX CONVENTIONS, which every window below depends on:
#   * a DECISION carries the PRE-increment move counter (meta["move"] == the index of the
#     observation it was taken on), and `EventTracker.observe` stamps that same index;
#   * an ACTION event carries the POST-increment counter, so the action taken AT decision m
#     is stamped m + 1 and the action that ENDS turn T is stamped with the first index at
#     which turn T + 1 is observed.
# Hence: the actions of turn T occupy the move range (first_move[T], first_move[T + 1]],
# exclusive-low / inclusive-high, which is the form every window uses.

def _v22_context(record):
    """_v21_context plus the v2.2 timeline: which seat owns each turn, where each turn
    starts, and each seat's per-select engine-state facts (prizes / locks)."""
    context = _v21_context(record)
    turn_owner = {int(turn): int(seat)
                  for turn, seat in (record.get("turn_owner") or {}).items()}
    owned_turns = {0: [], 1: []}
    for turn in sorted(turn_owner):
        owned_turns[turn_owner[turn]].append(turn)
    facts = record.get("decision_facts") or {0: [], 1: []}
    facts = {int(seat): list(entries) for seat, entries in facts.items()}
    context.update({
        "turn_owner": turn_owner,
        "turn_first_move": {int(turn): int(move) for turn, move
                            in (record.get("turn_first_move") or {}).items()},
        "owned_turns": owned_turns,
        # observed turns in order, so "where does the turn AFTER this one begin" never
        # depends on the turn counter being gapless
        "turn_order": sorted(int(turn) for turn
                             in (record.get("turn_first_move") or {})),
        "facts": facts,
        # each seat's facts are appended in move order -> "the latest at or before this
        # decision" is a bisect over these move columns
        "fact_moves": {seat: [entry[0] for entry in entries]
                       for seat, entries in facts.items()},
        "fact_prizes": {seat: _v22_prize_eras(
            entries, exact=record.get("prize_labels") == "truth")
            for seat, entries in facts.items()}})
    return context


def _v22_prize_eras(facts, exact=False):
    """Per fact, the seat's prize pile contents -- or None where they are not deducible.

    A prize pile only ever SHRINKS, and only when a prize is taken, so a run of consecutive
    selects with the same prize COUNT (public, on every observation) is a run with the same
    prize CONTENTS. One deducible select therefore pins the whole run. Two deductions inside
    one run that disagree would mean the pile changed without the count changing: the run is
    dropped rather than trusted.

    `exact=True` (--prize-labels truth) skips all of that: every fact already carries the
    engine's authoritative contents for ITS OWN decision, so there is nothing to propagate
    and nothing that a count-invariant prize swap can invalidate."""
    if exact:
        return [entry[3] for entry in facts]
    resolved = [None] * len(facts)
    start = 0
    while start < len(facts):
        stop = start
        while stop + 1 < len(facts) and facts[stop + 1][2] == facts[start][2]:
            stop += 1
        deduced = {entry[3] for entry in facts[start:stop + 1] if entry[3] is not None}
        if len(deduced) == 1:
            pinned = deduced.pop()
            for position in range(start, stop + 1):
                resolved[position] = pinned
        start = stop + 1
    return resolved


def _v22_windows(context, meta, mover):
    """The three v2.2 windows for ONE decision, as (actor, low, high] move ranges:

      A  the MOVER's remaining actions in the owned turn it is deciding inside. When the
         mover does NOT own this turn (a forced off-turn select) the turn has already ended
         for it, so the window is (m, m] -- EMPTY BUT VALID, which is the point of the
         rebuild: no horizon lands on the wrong side.
      B  everything the OPPONENT does between the end of that turn and the start of the
         mover's next owned turn.
      C  the mover's next owned turn, start to finish.

    A window whose end the game never reached is None (-> masked). Ownership comes from the
    MAIN-menu marker in the select stream; a turn with no marker at all (setup) leaves A
    masked rather than guessing."""
    turn, move = meta["turn"], meta["move"]
    first_move = context["turn_first_move"]
    owner = context["turn_owner"].get(turn)
    # end of the mover's current owned turn (or `move` itself when it owns no turn here)
    end = _v22_turn_end(context, turn) if owner == mover else move
    windows = [None, None, None]
    if owner is not None:
        windows[0] = (mover, move, end)
    next_turn = None
    owned = context["owned_turns"].get(mover) or []
    position = bisect.bisect_right(owned, turn)
    if position < len(owned) and owned[position] in first_move:
        next_turn = owned[position]
        start = first_move[next_turn]
        windows[1] = (1 - mover, end, start)
        windows[2] = (mover, start, _v22_turn_end(context, next_turn))
    return {"windows": windows, "next_turn": next_turn}


def _v22_turn_end(context, turn):
    """The move index of the last action of `turn` -- i.e. where the NEXT observed turn
    begins. V22_MOVE_INFINITY when the game ended inside this turn (its remaining actions
    are then all of them, which is what the window should cover)."""
    order = context["turn_order"]
    position = bisect.bisect_right(order, turn)
    if position >= len(order):
        return V22_MOVE_INFINITY
    return context["turn_first_move"][order[position]]


def _v22_window_counts(index, move_key, card_key, actor, low, high):
    """((card id, count class), ...) for one actor's cards in a (low, high] move range."""
    moves = index[move_key][actor]
    start = bisect.bisect_right(moves, low)
    stop = bisect.bisect_right(moves, high)
    if stop <= start:
        return ()
    counts = Counter(index[card_key][actor][start:stop])
    return tuple((card_id, min(count, V22_COUNT_CLASSES - 1))
                 for card_id, count in sorted(counts.items()) if card_id)


def _v22_window_attack(index, actor, low, high):
    """The attack `actor` DECLARED inside (low, high], 0 = none."""
    moves = index["attack_moves"][actor]
    start = bisect.bisect_right(moves, low)
    stop = bisect.bisect_right(moves, high)
    return next((value for value in index["attack_ids"][actor][start:stop]
                 if 0 < value < ATTACK_VOCAB), 0)


def _v22_window_retreat(index, actor, low, high):
    moves = index["retreat_moves"][actor]
    return float(bisect.bisect_right(moves, low) < bisect.bisect_right(moves, high))


def _v22_latest_prizes(context, seat, move):
    """That seat's prize pile as of `move`, read at its own latest select at or before it
    (for the DECIDING seat that IS the select being labelled). None -> masked."""
    moves = context["fact_moves"].get(seat) or []
    position = bisect.bisect_right(moves, move) - 1
    return context["fact_prizes"][seat][position] if position >= 0 else None


def _v22_token_labels(meta, events, context):
    """Per board token: the retained KO clocks / damage windows / still-in-play, plus the
    TYPED ATTACHMENTS -- which energy cards and tools are attached to that same physical
    card at +1 / +2 turns, as {0,1,2,3+} count classes per attachment id.

    Stored SPARSELY (a validity mask plus the non-zero cells): the dense form is
    [18, 2, 47] per decision, which is ~70 MB an iteration of almost entirely zeros.
    `collate` expands it into the minibatch."""
    board = (meta.get("board") or [])[:MAX_BOARD_TOKENS]
    present = np.full(MAX_BOARD_TOKENS, -1.0, dtype=np.float32)
    attach_mask = np.zeros((MAX_BOARD_TOKENS, V22_ATTACH_HORIZONS), dtype=bool)
    attach_items = []
    for horizon in (1, 2):
        snapshot = context["snapshots"].get(meta["turn"] + horizon)
        if snapshot is None:                        # game ended first -> masked
            continue
        in_play = snapshot["present"]
        attachments = snapshot.get("attachments") or {}
        for index, (serial, _owner, _hp) in enumerate(board):
            if horizon == 2:
                present[index] = 1.0 if serial in in_play else 0.0
            if serial not in in_play:
                continue                            # card gone: nothing is attached TO it
            attach_mask[index, horizon - 1] = True
            for card_id, count in Counter(attachments.get(serial) or ()).items():
                column = ATTACHMENT_INDEX.get(card_id)
                if column is not None:
                    attach_items.append((index, horizon - 1, column,
                                         min(count, V22_COUNT_CLASSES - 1)))
    return {"v21_ko": _ko_clock_labels(meta, events, context["index"]),
            "v21_damage": _damage_labels(meta, events, context["index"]),
            "v21_present": present,
            "v22_attach_mask": attach_mask,
            "v22_attach_items": attach_items}


def _v22_side_labels(meta, events, context, mover):
    """The retained SideFuture targets (prize / deck-out clocks, board composition, hand
    sizes, opponent hand / deck / active) minus the cut `side_energy`, plus:

    my_prizes / opp_prizes  the ids in each prize pile -- mine exactly as of this select,
                            theirs as of their own latest select at or before it (the same
                            convention the opponent-hand label already uses).
    stadium                 the stadium card id in play at +1 / +2 (0 = none).
    future_locks            the 10 restriction bits at the START of the mover's next MAIN
                            select run, i.e. the first MAIN select of window C's turn."""
    labels = _v21_side_labels(meta, events, context, mover)
    labels.pop("v21_side_energy", None)                  # CUT in v22
    stadium = np.full(V22_STADIUM_HORIZONS, -1, dtype=np.int64)
    for horizon in (1, 2):
        snapshot = context["snapshots"].get(meta["turn"] + horizon)
        if snapshot is not None:
            card_id = int(snapshot.get("stadium") or 0)
            stadium[horizon - 1] = card_id if 0 <= card_id < CARD_VOCAB else 0
    next_turn = _v22_windows(context, meta, mover)["next_turn"]
    locks = None
    if next_turn is not None:
        for _move, turn, _count, _prizes, fact_locks, main in context["facts"][mover]:
            if turn == next_turn and main and fact_locks is not None:
                locks = fact_locks
                break
    labels.update({
        "v22_my_prize_ids": _v22_latest_prizes(context, mover, meta["move"]),
        "v22_opp_prize_ids": _v22_latest_prizes(context, 1 - mover, meta["move"]),
        "v22_stadium": stadium,
        "v22_locks": (np.full(V22_LOCK_BITS, -1.0, dtype=np.float32) if locks is None
                      else locks)})
    return labels


def _v22_action_labels(meta, events, context, mover):
    """Per WINDOW (A / B / C, see _v22_windows), for the side the window belongs to: cards
    played and abilities used as {0,1,2,3+} count classes, the attack declared (0 = none)
    and whether it retreated -- plus the retained rest-of-game per-card use integrals.

    Sets with counts, no ordering: sequencing knowledge is search's job, not a label's."""
    index = context["index"]
    plan = _v22_windows(context, meta, mover)
    mask = np.zeros(V22_WINDOWS, dtype=bool)
    attack = np.full(V22_WINDOWS, -1, dtype=np.int64)
    retreat = np.full(V22_WINDOWS, -1.0, dtype=np.float32)
    plays, abilities = [], []
    for slot, window in enumerate(plan["windows"]):
        if window is None or index is None:
            plays.append(())
            abilities.append(())
            continue
        actor, low, high = window
        mask[slot] = True
        plays.append(_v22_window_counts(index, "play_moves", "play_cards",
                                        actor, low, high))
        abilities.append(_v22_window_counts(index, "ability_moves", "ability_cards",
                                            actor, low, high))
        attack[slot] = _v22_window_attack(index, actor, low, high)
        retreat[slot] = _v22_window_retreat(index, actor, low, high)
    if index is not None:
        use_ids = [list(index["use_suffix"][player][
            bisect.bisect_right(index["play_moves"][player], meta["move"])])
            for player in (mover, 1 - mover)]
    else:
        use_ids = [sorted({event["card"] for event in events
                           if event["kind"] == "play" and event["player"] == player
                           and event["move"] > meta["move"]})
                   for player in (mover, 1 - mover)]
    return {"v22_play_items": plays, "v22_ability_items": abilities,
            "v22_attack": attack, "v22_retreat": retreat,
            "v22_window_mask": mask,
            "v21_my_use_ids": use_ids[0], "v21_opp_use_ids": use_ids[1]}


def _v22_labels(meta, events, context, mover, modules):
    entry = {}
    if "token" in modules:
        entry.update(_v22_token_labels(meta, events, context))
    if "side" in modules:
        entry.update(_v22_side_labels(meta, events, context, mover))
    if "action" in modules:
        entry.update(_v22_action_labels(meta, events, context, mover))
    return entry


_CARD_ACTIONS = {}


def _card_actions(card_id):
    """(attackIds in card-table order, how many skills the card has) -- what a board row's
    ATTACK_SLOTS and SKILL_SLOTS columns mean. Pure identity lookup (cached: the label scan
    hits it once per board token per decision); no cost or energy-type arithmetic."""
    actions = _CARD_ACTIONS.get(card_id)
    if actions is None:
        from src.cards import get_card
        card = get_card(int(card_id)) or {}
        actions = _CARD_ACTIONS[card_id] = (
            tuple(attack_id for attack_id in (card.get("attacks") or [])[:ATTACK_SLOTS]
                  if attack_id),
            len(card.get("skills") or []))
    return actions


def _fill_payability_row(target, card_id, attack_offers, ability_offered):
    """ONE board row's columns: [attack 0, attack 1 | ability 0, ability 1], -1 = masked.

    attack_offers: the attackIds the engine offered while THIS Pokemon was the active, or
      None when it was not the active at all (its attack columns then stay masked -- a
      benched Pokemon is not offered attacks, so a 0 there would teach nothing).
    ability_offered: whether the engine offered an ABILITY hosted by this Pokemon, or None
      when it was not in play (masked).

    An ABILITY option names only its HOST (area + index -- cg.api.OptionType.ABILITY),
    never WHICH of the host's skills it activates. A host with one skill is therefore
    unambiguous; a host with two is not, and both of its ability columns stay masked."""
    attacks, skill_count = _card_actions(card_id)
    if attack_offers is not None:
        for slot, attack_id in enumerate(attacks):
            target[slot] = 1.0 if attack_id in attack_offers else 0.0
    if ability_offered is not None and skill_count == 1:
        target[ATTACK_SLOTS] = 1.0 if ability_offered else 0.0


def _main_offer_context(record):
    """The per-GAME lookup the payability labels share: each seat's MAIN-select captures in
    move order, plus (mover, move) -> that seat's index into them."""
    by_mover = {0: [], 1: []}
    for capture in record.get("main_offers") or []:
        by_mover[capture[1]].append(capture)
    return {"by_mover": by_mover,
            "position": {(mover, capture[0]): position
                         for mover, captures in by_mover.items()
                         for position, capture in enumerate(captures)}}


def _payability_labels(meta, context, mover):
    """The three payability targets for ONE decision, each [PAYABLE_BOARD_SLOTS,
    PAYABLE_SLOTS] over MY in-play Pokemon (encoder token order), -1 = masked:

    payable_now      what the engine offered for that Pokemon AT THIS select
    payable_next     ...at my NEXT MAIN select, while that same physical card is still
                     there (still the active for the attack columns, still in play for the
                     ability ones)
    payable_horizon  ...at ANY of my next 3 MAIN selects. Attacks count only while THAT
                     serial was the active (which is what teaches bench energy math: attach
                     to the bench, promote, attack); abilities count wherever it sat.

    All three exist only for a decision taken AT a MAIN select -- the only place the engine
    offers actions, hence the only place the labels exist. payable_horizon is masked
    entirely when the game ends before the 3-select window completes."""
    now = np.full((PAYABLE_BOARD_SLOTS, PAYABLE_SLOTS), -1.0, dtype=np.float32)
    following = np.full((PAYABLE_BOARD_SLOTS, PAYABLE_SLOTS), -1.0, dtype=np.float32)
    horizon = np.full((PAYABLE_BOARD_SLOTS, PAYABLE_SLOTS), -1.0, dtype=np.float32)
    labels = {"payable_now": now, "payable_next": following,
              "payable_horizon": horizon}
    position = context["position"].get((mover, meta["move"]))
    if position is None:
        return labels                       # not a MAIN select (or a resigned game's last
                                            # decision, whose select was never answered)
    captures = context["by_mover"][mover]
    _move, _mover, active_serial, board_serials, board_card_ids, attacks, abilities = \
        captures[position]
    # Token alignment: the heads read BOARD TOKEN ROWS, and the encoder emits my active,
    # my bench, then the opponent's. The capture and meta["board"] are built from the same
    # observation in that order, so a mismatch means the row a label lands on is not the
    # Pokemon it describes -- mask instead.
    if tuple(serial for serial, owner, _hp in meta["board"]
             if owner == mover) != board_serials:
        return labels
    following_capture = captures[position + 1] if position + 1 < len(captures) else None
    next_serials = () if following_capture is None else following_capture[3]
    window = captures[position + 1:position + 1 + PAYABLE_HORIZON_SELECTS]
    for row in range(min(len(board_serials), PAYABLE_BOARD_SLOTS)):
        serial, card_id = board_serials[row], board_card_ids[row]
        is_active = serial == active_serial
        _fill_payability_row(now[row], card_id, attacks if is_active else None,
                             serial in abilities)
        if following_capture is not None:
            _fill_payability_row(
                following[row], card_id,
                following_capture[5] if is_active
                and following_capture[2] == active_serial else None,
                (serial in following_capture[6]) if serial in next_serials else None)
        if len(window) == PAYABLE_HORIZON_SELECTS:
            _fill_payability_row(
                horizon[row], card_id,
                {attack_id for capture in window if capture[2] == serial
                 for attack_id in capture[5]},
                any(serial in capture[6] for capture in window))
    return labels


def _v21_labels(meta, events, context, mover, modules):
    entry = {}
    if "token" in modules:
        entry.update(_v21_token_labels(meta, events, context))
    if "side" in modules:
        entry.update(_v21_side_labels(meta, events, context, mover))
    if "action" in modules:
        entry.update(_v21_action_labels(meta, events, context, mover))
    return entry


def _v21_cross_checks(meta, events, context, mover, entry, stats):
    """Spec layer 3 (the REMOVED heads, kept as assertions) plus the event-dependent half
    of layer 2. Every one needs a fresh event scan, so assemble runs these on a sampled
    slice of decisions rather than all of them. Raises AssertionError on a violation --
    except the supporter relation, which is a monitored RATE (a supporter can also reach
    the discard by an effect rather than by being played, so it is not strictly forced)."""
    index = context["index"]
    if "v21_ko" in entry and meta["active_serial"] is not None:
        # survives == (KO-turn > 1)
        survives = _survives_label(meta, events, mover, index)
        ko_delta = None
        if index is not None:
            for event_move, event_turn, _event_hp, knocked_out in \
                    index["left"].get((mover, meta["active_serial"]), ()):
                if event_move > meta["move"] and knocked_out:
                    ko_delta = event_turn - meta["turn"]
                    break
        else:
            for event in events:
                if (event["kind"] == "left" and event["player"] == mover
                        and event["serial"] == meta["active_serial"]
                        and event["move"] > meta["move"] and _is_ko(events, event)):
                    ko_delta = event["turn"] - meta["turn"]
                    break
        assert (survives > 0.5) == (ko_delta is None or ko_delta > 1), \
            f"survives {survives} vs KO delta {ko_delta} (move {meta['move']})"
    if "v21_play_ids" in entry:
        for horizon in range(V21_HORIZONS):
            turn = meta["turn"] + horizon
            side = _horizon_side(context, turn, mover, horizon)
            window = [event for event in context["by_turn"].get(turn, ())
                      if event["player"] == side
                      and (horizon > 0 or event["move"] > meta["move"])]
            played = [event["card"] for event in window if event["kind"] == "play"
                      and event["card"]]
            labelled = entry["v21_play_ids"][horizon]
            # counts == sum(plays): the deleted counts head counted play EVENTS and the
            # plays head is a SET, so the forced relation is |set| <= |events| with the
            # zero case exact.
            assert set(labelled) <= set(played), "play label outside the play events"
            assert len(labelled) <= len(played) and bool(labelled) == bool(played), \
                f"counts/plays disagree at horizon {horizon}"
            # attack "none" iff no attack was DECLARED that turn, and -- from the
            # independent engine ATTACK log -- a declared attack really resolved. That log
            # is "since that player's last selection", so it can surface one turn late.
            declared = any(event["kind"] == "declare_attack" for event in window)
            if entry["v21_action_mask"][horizon]:
                assert (int(entry["v21_attack"][horizon]) != 0) == declared, \
                    f"attack label {entry['v21_attack'][horizon]} vs declared={declared}"
                label = int(entry["v21_attack"][horizon])
                logged = [event for later in (turn, turn + 1)
                          for event in context["by_turn"].get(later, ())
                          if event["kind"] == "attack" and event["player"] == side]
                assert not label or any(event["attack"] == label for event in logged), \
                    f"declared attack {label} never appears in the engine ATTACK log"
        supporter = _supporter_label(meta, context.get("snapshot_list") or [], mover,
                                     by_turn=context["snapshots"])
        if supporter > 0:
            stats["v21_check_supporter_n"] += 1
            if supporter not in entry["v21_play_ids"][1]:
                stats["v21_check_supporter_bad"] += 1


def _v22_cross_checks(meta, events, context, mover, entry, stats):
    """The v2.2 layer-3 checks: every window's play set, ability set, attack and retreat
    re-derived by a RAW SCAN of the event stream over that window's move range, against the
    labels the bisected index produced. Plus the window algebra itself (disjoint, ordered,
    correctly attributed) and the prize residual's own tripwires. Raises AssertionError."""
    if "v21_ko" in entry and meta["active_serial"] is not None and context["index"]:
        survives = _survives_label(meta, events, mover, context["index"])
        ko_delta = None
        for event_move, event_turn, _hp, knocked_out in \
                context["index"]["left"].get((mover, meta["active_serial"]), ()):
            if event_move > meta["move"] and knocked_out:
                ko_delta = event_turn - meta["turn"]
                break
        assert (survives > 0.5) == (ko_delta is None or ko_delta > 1), \
            f"survives {survives} vs KO delta {ko_delta} (move {meta['move']})"
    if "v22_play_items" in entry:
        plan = _v22_windows(context, meta, mover)
        bounds = []
        for slot, window in enumerate(plan["windows"]):
            if window is None:
                assert not bool(entry["v22_window_mask"][slot]), \
                    f"window {slot} has no bounds but is not masked"
                continue
            assert bool(entry["v22_window_mask"][slot]), \
                f"window {slot} has bounds but is masked"
            actor, low, high = window
            bounds.append((slot, low, high))
            assert actor == (1 - mover if slot == 1 else mover), \
                f"window {slot} attributed to the wrong seat"
            for key, kind in (("v22_play_items", "play"),
                              ("v22_ability_items", "ability")):
                scanned = Counter(event["card"] for event in events
                                  if event["kind"] == kind and event["player"] == actor
                                  and low < event["move"] <= high and event["card"])
                expected = {card_id: min(count, V22_COUNT_CLASSES - 1)
                            for card_id, count in scanned.items()}
                assert dict(entry[key][slot]) == expected, \
                    f"{kind} counts disagree with a raw scan of window {slot}"
            declared = [event["attack"] for event in events
                        if event["kind"] == "declare_attack" and event["player"] == actor
                        and low < event["move"] <= high
                        and 0 < event["attack"] < ATTACK_VOCAB]
            assert int(entry["v22_attack"][slot]) == (declared[0] if declared else 0), \
                f"attack label disagrees with a raw scan of window {slot}"
            assert float(entry["v22_retreat"][slot]) == float(any(
                event["kind"] == "retreat" and event["player"] == actor
                and low < event["move"] <= high for event in events)), \
                f"retreat label disagrees with a raw scan of window {slot}"
        for (left_slot, _low, left_high), (right_slot, right_low, _high) in \
                zip(bounds, bounds[1:]):
            assert left_high <= right_low, \
                f"windows {left_slot}/{right_slot} overlap ({left_high} > {right_low})"
    if "v22_my_prize_ids" in entry:
        # The prize labels are a RESIDUAL derivation (decklist minus every placed card), so
        # both their closure rate and their containment are monitored.
        for side, seat in (("my", mover), ("opp", 1 - mover)):
            stats[f"v22_check_{side}_prize_n"] += 1
            ids = entry[f"v22_{side}_prize_ids"]
            if ids is None:
                stats[f"v22_check_{side}_prize_masked"] += 1
                continue
            assert len(ids) <= 6, f"{side} prize residual holds {len(ids)} cards"
            assert set(ids) <= set(context["deck_ids"][seat]), \
                f"{side} prize residual holds a card outside that seat's decklist"


V21_CHECK_STRIDE = 37     # label-time cross-checks run on every Nth decision


def build_labels(record, heads="family1", aux_labels=True,
                 v21_modules=("token", "side", "action"), v21_checks=True,
                 payability=False, stats=None):
    """The aux LABELS of one finished game, as a list parallel to record["decisions"].

    Labels are a pure function of the finished game's event / select record -- no policy,
    no value, no cross-game state -- so this runs in the GENERATION WORKER (2026-07-31
    pipeline rework) and the main process only concatenates. The label VALUES are exactly
    what `_assemble_legacy` produced inline; parity_v22_pipeline.py is the gate.

    Returns [] when nothing is labelled (the fast path for policy+value-only runs)."""
    if stats is None:
        stats = Counter()
    aux_v21 = aux_labels and heads == "v21"
    aux_v22 = aux_labels and heads in ("v22", "v23", "v24", "v25", "v26")
    aux_v23 = aux_labels and heads in ("v23", "v24", "v25", "v26")
    if not (payability or aux_labels):
        return []
    events = record.get("events") or []
    snapshots = record.get("snapshots") or []
    label_context = None
    if aux_v21:
        label_context = _v21_context(record)
    elif aux_v22:
        label_context = _v22_context(record)
    # v23 = v22 + four head groups; its own per-game index is built beside the v22 one and
    # neither reads the other, so --heads v22 is bit-for-bit what it always was.
    v23_context = aux_head_labels.build_context(record) if aux_v23 else None
    offer_context = _main_offer_context(record) if payability else None
    entries = []
    for position, decision in enumerate(record["decisions"]):
        mover, meta = decision[5], decision[9]
        entry = {}
        if offer_context is not None:
            entry.update(_payability_labels(meta, offer_context, mover))
        if aux_v21 or aux_v22:
            builder = _v21_labels if aux_v21 else _v22_labels
            entry.update(builder(meta, events, label_context, mover, v21_modules))
            # Which SEAT decided. `opp_deck` is only an honest prediction for seat 0, whose
            # opponent is drawn from the field; seat 1's "opponent deck" is the fixed
            # --focus-deck and therefore a game constant (2026-08-06 audit measured the
            # trained head firing exactly 24.00 ids/row -- the focus list's distinct-id
            # count -- at recall 0.9999 on mover==1 rows).
            entry["v22_mover"] = mover
            # Sampled cross-checks (spec layer 3). The stride runs PER GAME here rather than
            # over the whole block -- the worker cannot know its position in one -- so the
            # sampled subset differs from the old placement; the checks themselves do not.
            if v21_checks and aux_v21 and position % V21_CHECK_STRIDE == 0:
                _v21_cross_checks(meta, events, label_context, mover, entry, stats)
            elif v21_checks and aux_v22 and position % V21_CHECK_STRIDE == 0:
                # COUNTED, not fatal (2026-08-07): a sampled diagnostic must never kill the
                # trainer -- this assert ("window 1 has bounds but is masked", a rare
                # end-of-game turn state) crashed d256_uniform repeatedly, and plausibly
                # explains d256_v6's six undiagnosed deaths too. The v23 tripwire
                # convention (count + preserve evidence) applies; the counter prints in
                # the iteration line's bad_* block, the detail goes to stderr (now
                # rotation-preserved per launch).
                try:
                    _v22_cross_checks(meta, events, label_context, mover, entry, stats)
                except AssertionError as check_error:
                    stats["v23_bad_v22_cross_check"] += 1
                    print(f"[v22-cross-check] move {meta.get('move')} turn "
                          f"{meta.get('turn')} mover {mover}: {check_error}",
                          file=sys.stderr, flush=True)
        elif aux_labels:
            entry.update({
                "my_clock": _clock_bucket(meta, events, mover),
                "opp_clock": _clock_bucket(meta, events, 1 - mover),
                "survives": _survives_label(meta, events, mover),
                "hand_flags": _hand_flags(meta["opp_hand_next_turn"])})
        if aux_v23:
            # ADDITIVE on top of the v22 block above (v23 = v22 + four head groups), and
            # deliberately OUTSIDE the if/elif chain so that chain is exactly as it was.
            entry.update(aux_head_labels.labels(meta, decision[6], v23_context, mover, stats))
            if v21_checks and position % V21_CHECK_STRIDE == 0:
                aux_head_labels.cross_checks(meta, v23_context, mover, entry, stats,
                                        decision[6])
        if heads == "full":
            entry.update(_full_labels(record, meta, events, snapshots, mover))
        entries.append(entry)
    return entries


def _full_labels(record, meta, events, snapshots, mover):
    """The legacy --heads full extras, lifted verbatim out of the assemble loop."""
    active_id = meta.get("opp_active_next_turn")
    return {
        "milestones": np.array(
            _milestone_buckets(meta, events, mover)
            + _milestone_buckets(meta, events, 1 - mover),
            dtype=np.int64),
        "ko_clock": _ko_clock_labels(meta, events),
        "damage": _damage_labels(meta, events),
        "deckout": np.array(_deckout_buckets(meta, events, mover),
                            dtype=np.int64),
        "hand_sizes": _hand_size_labels(meta, snapshots, mover),
        "supporter": _supporter_label(meta, snapshots, mover),
        "opp_active": int(active_id) if active_id
        and 0 < active_id < CARD_VOCAB else -1,
        "opp_hand_hot": _hand_multihot(meta["opp_hand_next_turn"]),
        "opp_deck_ids": record["deck_ids"][1 - mover],
        "my_use_ids": sorted({e["card"] for e in events
                              if e["kind"] == "used" and e["player"] == mover
                              and e["move"] > meta["move"]}),
        "opp_use_ids": sorted({e["card"] for e in events
                               if e["kind"] == "used" and e["player"] == 1 - mover
                               and e["move"] > meta["move"]}),
        "play_counts": np.array([
            min(sum(1 for e in events if e["kind"] == "used"
                    and e["player"] == mover and e["move"] > meta["move"]
                    and e["turn"] == meta["turn"]), 10) / 10.0,
            min(sum(1 for e in events if e["kind"] == "used"
                    and e["player"] == mover
                    and e["turn"] == meta["turn"] + 2), 10) / 10.0,
            min(sum(1 for e in events if e["kind"] == "used"
                    and e["player"] == 1 - mover
                    and e["turn"] == meta["turn"] + 1), 10) / 10.0,
        ], dtype=np.float32),
        "opp_board_ids": meta.get("opp_board_next_turn"),
    }


def assemble(game_records, gamma, lam, heads="family1", aux_labels=True,
             v21_modules=("token", "side", "action"), v21_checks=True, stats=None,
             payability=False, search=False):
    """Game records -> flat decision dicts: the policy/value columns (advantage, return,
    outcome -- computed HERE, they need the game's terminal reward) merged with the aux
    labels the WORKER already built and shipped as record["aux_entries"].

    A record without that key (an old worker, or the parity harness's OLD arm) is labelled
    here instead, so the function's output does not depend on where the labels were built.

    The two feature matrices are passed through in WHATEVER form the worker sent (dense
    ndarray or pack tuple); `collate` materialises them straight into the minibatch. See
    LAZY_UNPACK."""
    flat = []
    if stats is None:
        stats = Counter()
    for record in game_records:
        labels = record.get("aux_entries")
        if labels is None:
            labels = build_labels(record, heads=heads, aux_labels=aux_labels,
                                  v21_modules=v21_modules, v21_checks=v21_checks,
                                  payability=payability, stats=stats)
        else:
            stats.update(record.get("label_stats") or {})
        by_mover = {0: [], 1: []}
        for position, decision in enumerate(record["decisions"]):
            by_mover[decision[5]].append((position, decision))
        for mover, decisions in by_mover.items():
            if not decisions:
                continue
            reward = record["outcome"] if mover == 0 else -record["outcome"]
            values = np.array([d[1][8] for d in decisions], dtype=np.float32)
            advantages = gae(values, reward, gamma, lam)
            returns = np.clip(advantages + values, -1.0, 1.0)
            for (position, decision), advantage, target in zip(decisions, advantages,
                                                               returns):
                tokens, owners, zones, globals_, options, _m, chosen, logprob, value, \
                    _meta, card_ids = decision
                search_entry = None
                if search:
                    # --search-gen: the visit target rides in meta (see _append_decision).
                    # `searched` False everywhere is a valid EI iteration -- it just emits
                    # no policy loss -- so a missing key is a zero row, never an error.
                    search_entry = {
                        "searched": bool(_meta.get("searched")),
                        "search_target": _meta.get("search_target"),
                        "search_value": float(_meta.get("search_value") or 0.0),
                        "search_deep": bool(_meta.get("search_deep")),
                        "search_kl": float(_meta.get("search_kl") or 0.0)}
                entry = {"tokens": tokens if LAZY_UNPACK else _maybe_unpack(tokens),
                         "owners": owners,
                         "zones": zones, "globals": globals_,
                         "options": (options if LAZY_UNPACK
                                     else _maybe_unpack(options)), "chosen": chosen,
                         "card_ids": card_ids,
                         "logprob": logprob, "value": value,
                         "advantage": float(advantage), "return": float(target),
                         "outcome": reward}
                if labels:
                    entry.update(labels[position])
                if search_entry is not None:
                    entry.update(search_entry)
                flat.append(entry)
    return flat


def _assemble_legacy(game_records, gamma, lam, heads="family1", aux_labels=True,
             v21_modules=("token", "side", "action"), v21_checks=True, stats=None,
             payability=False):
    """The PRE-2026-07-31 assemble, frozen: every label built inline in the main process.

    Kept callable for one purpose -- parity_v22_pipeline.py (gate G0) runs a fixed seeded
    game set through this and through the new worker-side path and asserts the label arrays
    and the collated batch tensors are BYTE-IDENTICAL. Nothing else may call it."""
    flat = []
    check_counter = 0
    if stats is None:
        stats = Counter()
    for record in game_records:
        events = record.get("events") or []
        snapshots = record.get("snapshots") or []
        v21_context = _v21_context(record) if (aux_labels and heads == "v21") else None
        offer_context = _main_offer_context(record) if payability else None
        by_mover = {0: [], 1: []}
        for decision in record["decisions"]:
            by_mover[decision[5]].append(decision)
        for mover, decisions in by_mover.items():
            if not decisions:
                continue
            reward = record["outcome"] if mover == 0 else -record["outcome"]
            values = np.array([d[8] for d in decisions], dtype=np.float32)
            advantages = gae(values, reward, gamma, lam)
            returns = np.clip(advantages + values, -1.0, 1.0)
            for decision, advantage, target in zip(decisions, advantages, returns):
                tokens, owners, zones, globals_, options, _m, chosen, logprob, value, \
                    meta, card_ids = decision
                entry = {"tokens": tokens if LAZY_UNPACK else _maybe_unpack(tokens),
                         "owners": owners,
                         "zones": zones, "globals": globals_,
                         "options": (options if LAZY_UNPACK
                                     else _maybe_unpack(options)), "chosen": chosen,
                         "card_ids": card_ids,
                         "logprob": logprob, "value": value,
                         "advantage": float(advantage), "return": float(target),
                         "outcome": reward}
                if offer_context is not None:
                    entry.update(_payability_labels(meta, offer_context, mover))
                if aux_labels and heads == "v21":
                    entry.update(_v21_labels(meta, events, v21_context, mover,
                                             v21_modules))
                    check_counter += 1
                    if v21_checks and check_counter % V21_CHECK_STRIDE == 0:
                        _v21_cross_checks(meta, events, v21_context, mover, entry, stats)
                elif aux_labels:
                    entry.update({
                        "my_clock": _clock_bucket(meta, events, mover),
                        "opp_clock": _clock_bucket(meta, events, 1 - mover),
                        "survives": _survives_label(meta, events, mover),
                        "hand_flags": _hand_flags(meta["opp_hand_next_turn"])})
                if heads == "full":
                    active_id = meta.get("opp_active_next_turn")
                    entry.update({
                        "milestones": np.array(
                            _milestone_buckets(meta, events, mover)
                            + _milestone_buckets(meta, events, 1 - mover),
                            dtype=np.int64),
                        "ko_clock": _ko_clock_labels(meta, events),
                        "damage": _damage_labels(meta, events),
                        "deckout": np.array(_deckout_buckets(meta, events, mover),
                                            dtype=np.int64),
                        "hand_sizes": _hand_size_labels(meta, snapshots, mover),
                        "supporter": _supporter_label(meta, snapshots, mover),
                        "opp_active": int(active_id) if active_id
                        and 0 < active_id < CARD_VOCAB else -1,
                        "opp_hand_hot": _hand_multihot(meta["opp_hand_next_turn"]),
                        "opp_deck_ids": record["deck_ids"][1 - mover],
                        "my_use_ids": sorted({e["card"] for e in events
                                              if e["kind"] == "used"
                                              and e["player"] == mover
                                              and e["move"] > meta["move"]}),
                        "opp_use_ids": sorted({e["card"] for e in events
                                               if e["kind"] == "used"
                                               and e["player"] == 1 - mover
                                               and e["move"] > meta["move"]}),
                        "play_counts": np.array([
                            min(sum(1 for e in events if e["kind"] == "used"
                                    and e["player"] == mover
                                    and e["move"] > meta["move"]
                                    and e["turn"] == meta["turn"]), 10) / 10.0,
                            min(sum(1 for e in events if e["kind"] == "used"
                                    and e["player"] == mover
                                    and e["turn"] == meta["turn"] + 2), 10) / 10.0,
                            min(sum(1 for e in events if e["kind"] == "used"
                                    and e["player"] == 1 - mover
                                    and e["turn"] == meta["turn"] + 1), 10) / 10.0,
                        ], dtype=np.float32),
                        "opp_board_ids": meta.get("opp_board_next_turn"),
                    })
                flat.append(entry)
    return flat


def collate(batch, heads="family1", aux_labels=True, payability=False, search=False):
    if heads in ("v23", "v24", "v25", "v26") and aux_labels:
        return _add_search(_add_payability(
            aux_head_labels.collate_v23(_collate_v22(batch), batch), batch, payability),
            batch, search)
    if heads == "v22" and aux_labels:
        return _add_search(_add_payability(_collate_v22(batch), batch, payability),
                           batch, search)
    if heads == "v21" and aux_labels:
        return _add_search(_add_payability(_collate_v21(batch), batch, payability),
                           batch, search)
    if heads != "full":
        return _add_search(
            _add_payability(_collate_vectorized(batch, aux_labels=aux_labels),
                            batch, payability), batch, search)
    # The legacy 'full' path reads the two matrices as arrays in a dozen places; it is not
    # a training path today, so it materialises them up front instead of being rewritten.
    batch = [dict(d, tokens=_maybe_unpack(d["tokens"]),
                  options=_maybe_unpack(d["options"])) for d in batch]
    max_tokens = max(d["tokens"].shape[0] for d in batch)
    max_options = max(d["options"].shape[0] for d in batch)
    size = len(batch)
    out = {
        # tokens/options stay float16 host-side (RAM: they are the bulk of a batch)
        # and are cast to float32 on the GPU right before the forward.
        "tokens": torch.zeros(size, max_tokens, TOKEN_DIM, dtype=torch.float16),
        "owners": torch.zeros(size, max_tokens, dtype=torch.long),
        "zones": torch.zeros(size, max_tokens, dtype=torch.long),
        "padding": torch.ones(size, max_tokens, dtype=torch.bool),
        "globals": torch.zeros(size, GLOBAL_DIM),
        "options": torch.zeros(size, max_options, OPTION_DIM, dtype=torch.float16),
        "option_mask": torch.zeros(size, max_options, dtype=torch.bool),
        "chosen": torch.zeros(size, dtype=torch.long),
        "logprob": torch.zeros(size),
        "advantage": torch.zeros(size),
        "return": torch.zeros(size),
        "my_clock": torch.zeros(size, dtype=torch.long),
        "opp_clock": torch.zeros(size, dtype=torch.long),
        "survives": torch.zeros(size),
        "hand_flags": torch.zeros(size, HAND_FLAGS),
        "outcome": torch.zeros(size),
    }
    # v2 only: per-token card ids for the learned identity embedding. The key is ABSENT under
    # v1 so every downstream consumer (BatchPrefetcher's .to(), the training step) keeps
    # working untouched. 0 = no card, which is also what padding rows get -- the padding mask
    # hides them from attention anyway.
    if batch[0].get("card_ids") is not None:
        out["card_ids"] = torch.zeros(size, max_tokens, dtype=torch.long)
    if heads == "full":
        out.update({
            "milestones": torch.zeros(size, 2 * MILESTONES, dtype=torch.long),
            "ko_clock": torch.full((size, MAX_BOARD_TOKENS), -1, dtype=torch.long),
            "damage": torch.full((size, MAX_BOARD_TOKENS, DAMAGE_WINDOWS), -1.0),
            "deckout": torch.zeros(size, 2, dtype=torch.long),
            "hand_sizes": torch.full((size, 3), -1.0),
            "supporter": torch.full((size,), -1, dtype=torch.long),
            "opp_active": torch.full((size,), -1, dtype=torch.long),
            "opp_hand_hot": torch.zeros(size, CARD_VOCAB),
            "opp_hand_mask": torch.zeros(size, dtype=torch.bool),
            "opp_deck_hot": torch.zeros(size, CARD_VOCAB),
            "my_use_hot": torch.zeros(size, CARD_VOCAB),
            "opp_use_hot": torch.zeros(size, CARD_VOCAB),
            "play_counts": torch.zeros(size, 3),
            "opp_board_hot": torch.zeros(size, CARD_VOCAB),
            "opp_board_mask": torch.zeros(size, dtype=torch.bool),
        })
    for i, d in enumerate(batch):
        t, o = d["tokens"].shape[0], d["options"].shape[0]
        out["tokens"][i, :t] = torch.from_numpy(d["tokens"])
        out["owners"][i, :t] = torch.from_numpy(d["owners"].astype(np.int64))
        out["zones"][i, :t] = torch.from_numpy(d["zones"].astype(np.int64))
        out["padding"][i, :t] = False
        if "card_ids" in out:
            out["card_ids"][i, :t] = torch.from_numpy(d["card_ids"].astype(np.int64))
        out["globals"][i] = torch.from_numpy(d["globals"])
        out["options"][i, :o] = torch.from_numpy(d["options"])
        out["option_mask"][i, :o] = True
        out["chosen"][i] = int(d["chosen"])
        out["logprob"][i] = float(d["logprob"])
        out["advantage"][i] = float(d["advantage"])
        out["return"][i] = float(d["return"])
        out["outcome"][i] = float(d["outcome"])
        out["my_clock"][i] = int(d["my_clock"])
        out["opp_clock"][i] = int(d["opp_clock"])
        out["survives"][i] = float(d["survives"])
        out["hand_flags"][i] = torch.from_numpy(d["hand_flags"])
        if heads == "full":
            out["milestones"][i] = torch.from_numpy(d["milestones"])
            out["ko_clock"][i] = torch.from_numpy(d["ko_clock"])
            out["damage"][i] = torch.from_numpy(d["damage"])
            out["deckout"][i] = torch.from_numpy(d["deckout"])
            out["hand_sizes"][i] = torch.from_numpy(d["hand_sizes"])
            out["supporter"][i] = int(d["supporter"])
            out["opp_active"][i] = int(d["opp_active"])
            if d["opp_hand_hot"] is not None:
                out["opp_hand_hot"][i] = torch.from_numpy(d["opp_hand_hot"])
                out["opp_hand_mask"][i] = True
            for card_id in d["opp_deck_ids"]:
                if 0 < card_id < CARD_VOCAB:
                    out["opp_deck_hot"][i, card_id] = 1.0
            for key, hot in (("my_use_ids", "my_use_hot"),
                             ("opp_use_ids", "opp_use_hot")):
                for card_id in d[key]:
                    if 0 < card_id < CARD_VOCAB:
                        out[hot][i, card_id] = 1.0
            out["play_counts"][i] = torch.from_numpy(d["play_counts"])
            if d["opp_board_ids"] is not None:
                out["opp_board_mask"][i] = True
                for card_id in d["opp_board_ids"]:
                    if 0 < card_id < CARD_VOCAB:
                        out["opp_board_hot"][i, card_id] = 1.0
    return _add_payability(out, batch, payability)


def _add_payability(out, batch, payability):
    """The three --attack-aux-weight target tensors, on top of whichever --heads set the
    batch already carries (they are independent of it). No-op when the flag is off."""
    if payability:
        for key in ("payable_now", "payable_next", "payable_horizon"):
            out[key] = _v21_stack(batch, key)
    return out


def _add_search(out, batch, search):
    """The three --search-gen target tensors, on top of whatever --heads set the batch
    already carries. No-op when the flag is off, so every other run collates as before.

    `search_target` is padded to the batch's option width with zeros and masked by
    `searched`: an unsearched row contributes no policy loss at all, which is exactly
    'policy targets only from searched decisions' (addendum 15)."""
    if not search:
        return out
    size = len(batch)
    width = int(out["options"].shape[1])
    target = torch.zeros(size, width)
    searched = torch.zeros(size, dtype=torch.bool)
    values = torch.zeros(size)
    deep = torch.zeros(size, dtype=torch.bool)
    for position, decision in enumerate(batch):
        values[position] = float(decision.get("search_value") or 0.0)
        if not decision.get("searched"):
            continue
        row = decision.get("search_target") or []
        if not row:
            continue
        searched[position] = True
        deep[position] = bool(decision.get("search_deep"))
        target[position, :len(row)] = torch.tensor(row[:width], dtype=torch.float32)
    out["search_target"] = target
    out["searched"] = searched
    out["search_value"] = values
    out["search_deep"] = deep
    return out


def _collate_vectorized(batch, aux_labels=True):
    """collate's family1/none path with the per-row torch copy loop replaced by numpy
    whole-batch/row-wise fills -- byte-identical output (parity-checked: dtypes, shapes,
    values), ~3x faster. aux_labels=False omits the aux target tensors entirely
    (policy+value-only runs never read them; saves the H2D transfer too)."""
    size = len(batch)
    token_lengths = np.array([dense_shape(d["tokens"])[0] for d in batch], dtype=np.int64)
    option_lengths = np.array([dense_shape(d["options"])[0] for d in batch], dtype=np.int64)
    max_tokens = int(token_lengths.max())
    max_options = int(option_lengths.max())

    # The feature blocks are written ROW-WISE straight into the padded output: the earlier
    # concatenate-then-scatter form moved every byte twice (once into the concatenation,
    # once into the destination) and this path is memory-bound, so one pass is ~1.6x
    # faster. Same destination, same casts, byte-identical result. Under LAZY_UNPACK the
    # row still arrives PACKED and `scatter_dense` writes it in place, so the dense
    # intermediate assemble used to build is never allocated at all.
    tokens = np.zeros((size, max_tokens, TOKEN_DIM), dtype=np.float16)
    owners = np.zeros((size, max_tokens), dtype=np.int64)
    zones = np.zeros((size, max_tokens), dtype=np.int64)
    options = np.zeros((size, max_options, OPTION_DIM), dtype=np.float16)
    card_ids = (np.zeros((size, max_tokens), dtype=np.int64)
                if batch[0].get("card_ids") is not None else None)
    for position, decision in enumerate(batch):
        length = int(token_lengths[position])
        scatter_dense(decision["tokens"], tokens[position, :length])
        owners[position, :length] = decision["owners"]
        zones[position, :length] = decision["zones"]
        scatter_dense(decision["options"],
                      options[position, :int(option_lengths[position])])
        if card_ids is not None:
            card_ids[position, :length] = decision["card_ids"]

    out = {
        "tokens": torch.from_numpy(tokens),
        "owners": torch.from_numpy(owners),
        "zones": torch.from_numpy(zones),
        "padding": torch.from_numpy(
            np.arange(max_tokens)[None, :] >= token_lengths[:, None]),
        "globals": torch.from_numpy(np.stack([d["globals"] for d in batch])),
        "options": torch.from_numpy(options),
        "option_mask": torch.from_numpy(
            np.arange(max_options)[None, :] < option_lengths[:, None]),
        "chosen": torch.from_numpy(
            np.array([d["chosen"] for d in batch], dtype=np.int64)),
        "logprob": torch.from_numpy(
            np.array([d["logprob"] for d in batch], dtype=np.float32)),
        "advantage": torch.from_numpy(
            np.array([d["advantage"] for d in batch], dtype=np.float32)),
        "return": torch.from_numpy(
            np.array([d["return"] for d in batch], dtype=np.float32)),
        "outcome": torch.from_numpy(
            np.array([d["outcome"] for d in batch], dtype=np.float32)),
    }
    if card_ids is not None:
        out["card_ids"] = torch.from_numpy(card_ids)
    if aux_labels:
        out["my_clock"] = torch.from_numpy(
            np.array([d["my_clock"] for d in batch], dtype=np.int64))
        out["opp_clock"] = torch.from_numpy(
            np.array([d["opp_clock"] for d in batch], dtype=np.int64))
        out["survives"] = torch.from_numpy(
            np.array([d["survives"] for d in batch], dtype=np.float32))
        out["hand_flags"] = torch.from_numpy(
            np.stack([d["hand_flags"] for d in batch]).astype(np.float32))
    return out


def _v21_multihot(batch, key, rows=None):
    """Per-decision card-id LISTS -> a (size[, rows], CARD_VOCAB) BOOL multi-hot. Bool, not
    float: the vocab targets are the RAM bulk of a v21 minibatch (and of its H2D copy);
    the losses cast the slice they use.

    Whole-batch scatter (the _collate_vectorized pattern): the batch's id lists are
    flattened once, the row index of each id comes from np.repeat over the list lengths,
    and one fancy-index assignment sets every bit. Same bits as the per-row loop -- a
    multi-hot is order- and duplicate-insensitive."""
    size = len(batch)
    shape = (size, CARD_VOCAB) if rows is None else (size, rows, CARD_VOCAB)
    lists = ([decision[key] or () for decision in batch] if rows is None
             else [card_ids or () for decision in batch for card_ids in decision[key]])
    lengths = np.fromiter(map(len, lists), dtype=np.int64, count=len(lists))
    total = int(lengths.sum())
    hot = np.zeros(shape[0] * (1 if rows is None else rows) * CARD_VOCAB, dtype=bool)
    if total:
        ids = np.fromiter(chain.from_iterable(lists), dtype=np.int64, count=total)
        row = np.repeat(np.arange(len(lists), dtype=np.int64), lengths)
        valid = (ids > 0) & (ids < CARD_VOCAB)
        if valid.all():                    # the normal case: no out-of-vocabulary id
            hot[row * CARD_VOCAB + ids] = True
        else:
            hot[row[valid] * CARD_VOCAB + ids[valid]] = True
    return torch.from_numpy(hot.reshape(shape))


def _v21_stack(batch, key):
    """np.stack over one label column, straight into a preallocated block: the label arrays
    all carry the same fixed shape and dtype, so np.stack's per-array asanyarray/promotion
    machinery is pure overhead. Same bytes, ~1.7x cheaper."""
    first = batch[0][key]
    out = np.empty((len(batch),) + first.shape, dtype=first.dtype)
    for position, decision in enumerate(batch):
        out[position] = decision[key]
    return torch.from_numpy(out)


def _collate_v21(batch):
    """The policy/value tensors (unchanged, vectorized) plus the v2.1 targets for whichever
    modules assemble produced labels for."""
    out = _collate_vectorized(batch, aux_labels=False)
    stacked = lambda key: _v21_stack(batch, key)                        # noqa: E731
    if "v21_ko" in batch[0]:                                          # TokenFuture
        out["v21_ko"] = stacked("v21_ko")
        out["v21_damage"] = stacked("v21_damage")
        out["v21_energy"] = stacked("v21_energy")
        out["v21_present"] = stacked("v21_present")
    if "v21_prize" in batch[0]:                                       # SideFuture
        out["v21_prize"] = stacked("v21_prize")
        out["v21_deckout"] = stacked("v21_deckout")
        out["v21_board_hot"] = _v21_multihot(batch, "v21_board_ids", rows=V21_BOARD_SLOTS)
        out["v21_board_mask"] = stacked("v21_board_mask")
        out["v21_side_energy"] = stacked("v21_side_energy")
        out["v21_hand_sizes"] = stacked("v21_hand_sizes")
        out["v21_opp_hand_hot"] = _v21_multihot(batch, "v21_opp_hand_ids")
        out["v21_opp_hand_mask"] = torch.from_numpy(
            np.array([d["v21_opp_hand_ids"] is not None for d in batch], dtype=bool))
        out["v21_opp_deck_hot"] = _v21_multihot(batch, "v21_opp_deck_ids")
        out["v21_opp_active"] = torch.from_numpy(
            np.array([d["v21_opp_active"] for d in batch], dtype=np.int64))
    if "v21_attack" in batch[0]:                                      # ActionFuture
        out["v21_play_hot"] = _v21_multihot(batch, "v21_play_ids", rows=V21_HORIZONS)
        out["v21_ability_hot"] = _v21_multihot(batch, "v21_ability_ids",
                                               rows=V21_HORIZONS)
        out["v21_attack"] = stacked("v21_attack")
        out["v21_retreat"] = stacked("v21_retreat")
        out["v21_action_mask"] = stacked("v21_action_mask")
        out["v21_my_use_hot"] = _v21_multihot(batch, "v21_my_use_ids")
        out["v21_opp_use_hot"] = _v21_multihot(batch, "v21_opp_use_ids")
        out["v21_h1_opponent"] = torch.from_numpy(
            np.array([bool(d.get("v21_h1_opponent", True)) for d in batch], dtype=bool))
    return out


def _v22_scatter_grid(cells, mask, shape):
    """Sparse count-class cells -> the dense int8 label grid the CE heads read.

    -1 everywhere, 0 wherever the mask says that row is VALID (class 0 = "that card was not
    played / attached", a real label rather than padding), then the observed cells. int8,
    not int64: dense over the 1268-card vocab this is the RAM bulk of a v2.2 minibatch, and
    the loss casts only the masked slice it uses."""
    grid = np.full(shape, -1, dtype=np.int8)
    grid[mask] = 0
    if cells:
        coordinates = np.array(cells, dtype=np.int64)
        grid[tuple(coordinates[:, column]
                   for column in range(coordinates.shape[1] - 1))] = \
            coordinates[:, -1].astype(np.int8)
    return torch.from_numpy(grid)


def _v22_window_grid(batch, key, mask, size):
    """(decision, window, card id) -> count class, from the per-window (id, class) lists."""
    return _v22_scatter_grid(
        [(position, window, card_id, count)
         for position, decision in enumerate(batch)
         for window, entries in enumerate(decision[key])
         for card_id, count in entries
         if 0 < card_id < CARD_VOCAB],
        mask, (size, V22_WINDOWS, CARD_VOCAB))


def _v22_attach_grid(batch, mask, size):
    """(decision, board token, horizon, attachment column) -> count class."""
    return _v22_scatter_grid(
        [(position,) + cell
         for position, decision in enumerate(batch)
         for cell in decision["v22_attach_items"]],
        mask, (size, MAX_BOARD_TOKENS, V22_ATTACH_HORIZONS, ATTACHMENT_VOCAB))


def _collate_v22(batch):
    """The policy/value tensors (unchanged, vectorized) plus the v2.2 targets."""
    out = _collate_vectorized(batch, aux_labels=False)
    stacked = lambda key: _v21_stack(batch, key)                        # noqa: E731
    size = len(batch)
    if "v22_mover" in batch[0]:
        out["v22_mover"] = torch.from_numpy(
            np.array([d["v22_mover"] for d in batch], dtype=np.int64))
    if "v21_ko" in batch[0]:                                          # TokenFuture
        out["v21_ko"] = stacked("v21_ko")
        out["v21_damage"] = stacked("v21_damage")
        out["v21_present"] = stacked("v21_present")
        out["v22_attach"] = _v22_attach_grid(
            batch, np.stack([d["v22_attach_mask"] for d in batch]), size)
    if "v21_prize" in batch[0]:                                       # SideFuture
        out["v21_prize"] = stacked("v21_prize")
        out["v21_deckout"] = stacked("v21_deckout")
        out["v21_board_hot"] = _v21_multihot(batch, "v21_board_ids", rows=V21_BOARD_SLOTS)
        out["v21_board_mask"] = stacked("v21_board_mask")
        out["v21_hand_sizes"] = stacked("v21_hand_sizes")
        out["v21_opp_hand_hot"] = _v21_multihot(batch, "v21_opp_hand_ids")
        out["v21_opp_hand_mask"] = torch.from_numpy(
            np.array([d["v21_opp_hand_ids"] is not None for d in batch], dtype=bool))
        out["v21_opp_deck_hot"] = _v21_multihot(batch, "v21_opp_deck_ids")
        out["v21_opp_active"] = torch.from_numpy(
            np.array([d["v21_opp_active"] for d in batch], dtype=np.int64))
        for side in ("my", "opp"):
            out[f"v22_{side}_prize_hot"] = _v21_multihot(batch, f"v22_{side}_prize_ids")
            out[f"v22_{side}_prize_mask"] = torch.from_numpy(np.array(
                [d[f"v22_{side}_prize_ids"] is not None for d in batch], dtype=bool))
        out["v22_stadium"] = stacked("v22_stadium")
        out["v22_locks"] = stacked("v22_locks")
    if "v22_attack" in batch[0]:                                      # ActionFuture
        window_mask = np.stack([d["v22_window_mask"] for d in batch])
        out["v22_play_counts"] = _v22_window_grid(batch, "v22_play_items",
                                                  window_mask, size)
        out["v22_ability_counts"] = _v22_window_grid(batch, "v22_ability_items",
                                                     window_mask, size)
        out["v22_attack"] = stacked("v22_attack")
        out["v22_retreat"] = stacked("v22_retreat")
        out["v22_window_mask"] = torch.from_numpy(window_mask)
        out["v21_my_use_hot"] = _v21_multihot(batch, "v21_my_use_ids")
        out["v21_opp_use_hot"] = _v21_multihot(batch, "v21_opp_use_ids")
    return out


def v22_batch_checks(batch):
    """The collated-batch invariants that survive the v2.2 rebuild, plus the new ones.
    Vectorized and always on under --v21-checks, exactly like v21_batch_checks."""
    if "v21_prize" in batch:
        prize = batch["v21_prize"].view(-1, 2, V21_PRIZE_K)
        assert bool(((prize >= 0) & (prize < NUM_TIMING_CLASSES)).all()), \
            "prize-turn class out of range"
        assert bool((prize[:, :, 1:] >= prize[:, :, :-1]).all()), \
            "turns_until_k_prizes must be non-decreasing in k"
        deck = batch["v21_opp_deck_hot"]
        assert not bool((batch["v21_opp_hand_hot"] & ~deck)[
            batch["v21_opp_hand_mask"]].any()), \
            "opponent hand label holds a card outside the opponent's deck"
        # The prize labels are a RESIDUAL of a decklist, so the seat-swap tripwire is the
        # sharpest check there is: their prizes must be drawable from their deck.
        assert not bool((batch["v22_opp_prize_hot"] & ~deck)[
            batch["v22_opp_prize_mask"]].any()), \
            "opponent prize label holds a card outside the opponent's deck"
        assert bool((batch["v22_opp_prize_hot"][batch["v22_opp_prize_mask"]]
                     .sum(dim=-1) <= 6).all()), "more than 6 distinct prize cards"
        locks = batch["v22_locks"]
        assert bool((((locks == 0) | (locks == 1)) | (locks < 0)).all()), \
            "future_locks label is not a bit"
        stadium = batch["v22_stadium"]
        assert bool(((stadium >= -1) & (stadium < CARD_VOCAB)).all()), \
            "stadium class out of range"
    if "v22_attack" in batch:
        mask = batch["v22_window_mask"]
        attack = batch["v22_attack"]
        assert bool(((attack[mask] >= 0) & (attack[mask] < ATTACK_VOCAB)).all()), \
            "attack class out of range"
        assert bool((attack[~mask] < 0).all()), "masked window carries an attack class"
        counts = batch["v22_play_counts"]
        assert bool((counts[~mask] < 0).all()), "masked window carries play labels"
        assert bool(((counts[mask] >= 0) & (counts[mask] < V22_COUNT_CLASSES)).all()), \
            "play count class out of range"
    if "v22_attach" in batch:
        attach = batch["v22_attach"]
        assert bool(((attach >= -1) & (attach < V22_COUNT_CLASSES)).all()), \
            "attachment count class out of range"
    if "v21_ko" in batch:
        _v21_damage_checks(batch)


def v21_batch_checks(batch):
    """Spec layer 2, on the collated batch: cheap, vectorized, and ALWAYS ON under
    --v21-checks (not just in the harness). Raises AssertionError on a violation.

    The event-dependent half of layer 2 (attack 'none' iff no ATTACK that turn, plays
    inside the play events) lives in `_v21_cross_checks` at assemble time, where the event
    stream is still around."""
    if "v21_prize" in batch:
        prize = batch["v21_prize"].view(-1, 2, V21_PRIZE_K)
        assert bool(((prize >= 0) & (prize < NUM_TIMING_CLASSES)).all()), \
            "prize-turn class out of range"
        assert bool((prize[:, :, 1:] >= prize[:, :, :-1]).all()), \
            "turns_until_k_prizes must be non-decreasing in k"
        assert bool(((batch["v21_deckout"] >= 0)
                     & (batch["v21_deckout"] < NUM_TIMING_CLASSES)).all()), \
            "deckout class out of range"
        deck = batch["v21_opp_deck_hot"]
        hand_mask = batch["v21_opp_hand_mask"]
        # Seat-swap tripwire: everything we claim about the OPPONENT must be drawable from
        # the OPPONENT's decklist. An inverted side would light this up instantly.
        assert not bool((batch["v21_opp_hand_hot"] & ~deck)[hand_mask].any()), \
            "opponent hand label holds a card outside the opponent's deck"
        opponent_board = batch["v21_board_hot"][:, 2:] & ~deck.unsqueeze(1)
        assert not bool(opponent_board[batch["v21_board_mask"][:, 2:]].any()), \
            "opponent board label holds a card outside the opponent's deck"
        if "v21_play_hot" in batch:
            # Horizon 1 is the OPPONENT's turn on all but a handful of rows (forced
            # off-turn selects, and setup decisions where +1 is the first player's turn).
            # Restrict the tripwire to the rows where it holds rather than weakening it.
            opponent_turn = batch["v21_action_mask"][:, 1] & batch["v21_h1_opponent"]
            assert not bool((batch["v21_play_hot"][:, 1] & ~deck)[opponent_turn].any()), \
                "opponent play label holds a card outside the opponent's deck"
    if "v21_attack" in batch:
        mask = batch["v21_action_mask"]
        attack = batch["v21_attack"]
        assert bool(((attack[mask] >= 0) & (attack[mask] < ATTACK_VOCAB)).all()), \
            "attack class out of range"
        assert bool((attack[~mask] < 0).all()), "masked horizon carries an attack class"
        assert not bool(batch["v21_play_hot"][~mask].any()), \
            "masked horizon carries play labels"
    if "v21_ko" in batch:
        _v21_damage_checks(batch)


def _v21_damage_checks(batch):
    """The KO-clock / damage-window relations of layer 2 (shared by v21 and v22, whose
    token labels for those two targets are the same derivation)."""
    board_width = batch["v21_damage"].shape[1]
    knocked_out = ((batch["v21_ko"] >= 0)
                   & (batch["v21_ko"] < NUM_TIMING_CLASSES - 1))[:, :board_width]
    damage = batch["v21_damage"]
    # KO bucket b means the KO lands within 2b turns, i.e. inside window w for any
    # b <= w+1, and the damage label then carries at least the KO'd Pokemon's last HP.
    # Exception: the damage label is CAPPED at the token's HP at decision time, and a
    # Pokemon can sit at 0 HP awaiting the KO's resolution -- such a token can only
    # ever label 0, which the last window identifies.
    has_cap = damage[:, :, DAMAGE_WINDOWS - 1] > 0
    for window in range(DAMAGE_WINDOWS):
        doomed = knocked_out & (batch["v21_ko"][:, :board_width] <= window + 1) \
            & (damage[:, :, window] >= 0) & has_cap
        assert bool((damage[:, :, window][doomed] > 0).all()), \
            f"token KO'd inside window {window} but zero damage label"
    both = (damage[:, :, :-1] >= 0) & (damage[:, :, 1:] >= 0)
    assert bool((damage[:, :, 1:][both] >= damage[:, :, :-1][both]).all()), \
        "damage windows must be non-decreasing"


class AuxHeads(torch.nn.Module):
    """Linear heads off the CLS context (family 1 of MY_MODEL_DESIGN.md): prize clocks
    both sides, active-survives, true-opponent-hand flags. Saved as aux_state; the trunk
    state_dict stays loadable everywhere unchanged."""

    def __init__(self, d_model=128):
        super().__init__()
        self.my_clock = torch.nn.Linear(d_model, NUM_TIMING_CLASSES)
        self.opp_clock = torch.nn.Linear(d_model, NUM_TIMING_CLASSES)
        self.active_survives = torch.nn.Linear(d_model, 1)
        self.opp_hand = torch.nn.Linear(d_model, HAND_FLAGS)


def aux_losses(aux, context, batch, stats, n):
    loss = torch.zeros((), device=context.device)
    for name, head in (("my_clock", aux.my_clock), ("opp_clock", aux.opp_clock)):
        logits = head(context)
        loss = loss + torch.nn.functional.cross_entropy(logits, batch[name])
        stats[f"aux_{name}"] += float((logits.argmax(dim=-1)
                                       == batch[name]).float().mean()) * n
    survives_mask = batch["survives"] >= 0
    if survives_mask.any():
        logits = aux.active_survives(context).squeeze(-1)[survives_mask]
        target = batch["survives"][survives_mask]
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        stats["aux_survives"] += float(((logits > 0).float() == target).float().mean()) * n
    hand_mask = batch["hand_flags"][:, 0] >= 0
    if hand_mask.any():
        logits = aux.opp_hand(context)[hand_mask]
        target = batch["hand_flags"][hand_mask]
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        stats["aux_hand"] += float(((logits > 0).float() == target).float().mean()) * n
    return loss


class AuxHeadsFull(torch.nn.Module):
    """The MY_MODEL_DESIGN.md head set (linear, on CLS context + per-token embeddings):
    milestone prize clocks (k-th next prize, both sides), per-token KO clocks
    (evolution-line persistent), deck-out clocks, hand-size trajectory, opponent's next
    supporter, opponent's next-turn active, full-vocab opponent hand, active-survives."""

    def __init__(self, d_model=128):
        super().__init__()
        self.milestones = torch.nn.Linear(d_model, 2 * MILESTONES * NUM_TIMING_CLASSES)
        self.ko_clock = torch.nn.Linear(d_model, NUM_TIMING_CLASSES)     # per token
        self.damage = torch.nn.Linear(d_model, DAMAGE_WINDOWS)           # per token
        self.deckout = torch.nn.Linear(d_model, 2 * NUM_TIMING_CLASSES)
        self.hand_sizes = torch.nn.Linear(d_model, 3)
        self.supporter = torch.nn.Linear(d_model, CARD_VOCAB)
        self.opp_active = torch.nn.Linear(d_model, CARD_VOCAB)
        self.opp_hand = torch.nn.Linear(d_model, CARD_VOCAB)
        self.opp_deck = torch.nn.Linear(d_model, CARD_VOCAB)
        self.my_use = torch.nn.Linear(d_model, CARD_VOCAB)
        self.opp_use = torch.nn.Linear(d_model, CARD_VOCAB)
        self.play_counts = torch.nn.Linear(d_model, 3)
        self.opp_board = torch.nn.Linear(d_model, CARD_VOCAB)
        self.active_survives = torch.nn.Linear(d_model, 1)


VOCAB_LOSS_SCALE = 1.0 / float(np.log(CARD_VOCAB))    # keep 1268-way CEs O(1) vs the rest


def _masked_ce(logits, labels, stats, key, n):
    mask = labels >= 0
    if not mask.any():
        return torch.zeros((), device=logits.device)
    logits, labels = logits[mask], labels[mask]
    stats[key] += float((logits.argmax(dim=-1) == labels).float().mean()) * n
    return torch.nn.functional.cross_entropy(logits, labels)


def aux_losses_full(aux, context, token_embeddings, batch, stats, n):
    size = context.shape[0]
    loss = torch.zeros((), device=context.device)

    milestone_logits = aux.milestones(context).view(size, 2 * MILESTONES,
                                                    NUM_TIMING_CLASSES)
    loss = loss + torch.nn.functional.cross_entropy(
        milestone_logits.reshape(-1, NUM_TIMING_CLASSES),
        batch["milestones"].reshape(-1))
    stats["aux_clock"] += float((milestone_logits[:, 0].argmax(dim=-1)
                                 == batch["milestones"][:, 0]).float().mean()) * n

    board_width = min(token_embeddings.shape[1], MAX_BOARD_TOKENS)
    ko_logits = aux.ko_clock(token_embeddings[:, :board_width])
    loss = loss + _masked_ce(ko_logits.reshape(-1, NUM_TIMING_CLASSES),
                             batch["ko_clock"][:, :board_width].reshape(-1),
                             stats, "aux_ko", n)

    damage_target = batch["damage"][:, :board_width]
    damage_mask = damage_target >= 0
    if damage_mask.any():
        damage_predicted = torch.sigmoid(aux.damage(token_embeddings[:, :board_width]))
        loss = loss + ((damage_predicted[damage_mask]
                        - damage_target[damage_mask]) ** 2).mean()
        stats["aux_damage"] += float((damage_predicted[damage_mask].detach()
                                      - damage_target[damage_mask]).abs().mean()
                                     * DAMAGE_CAP) * n

    deckout_logits = aux.deckout(context).view(size, 2, NUM_TIMING_CLASSES)
    loss = loss + torch.nn.functional.cross_entropy(
        deckout_logits.reshape(-1, NUM_TIMING_CLASSES), batch["deckout"].reshape(-1))
    stats["aux_deckout"] += float((deckout_logits.argmax(dim=-1)
                                   == batch["deckout"]).float().mean()) * n

    size_mask = batch["hand_sizes"] >= 0
    if size_mask.any():
        predicted = torch.sigmoid(aux.hand_sizes(context))[size_mask]
        target = batch["hand_sizes"][size_mask]
        loss = loss + ((predicted - target) ** 2).mean()
        stats["aux_handsize"] += float((predicted.detach() - target).abs().mean()
                                       * HAND_SIZE_CAP) * n

    loss = loss + VOCAB_LOSS_SCALE * _masked_ce(aux.supporter(context),
                                                batch["supporter"], stats,
                                                "aux_supporter", n)
    loss = loss + VOCAB_LOSS_SCALE * _masked_ce(aux.opp_active(context),
                                                batch["opp_active"], stats,
                                                "aux_active", n)

    hand_mask = batch["opp_hand_mask"]
    if hand_mask.any():
        hand_logits = aux.opp_hand(context)[hand_mask]
        hand_target = batch["opp_hand_hot"][hand_mask]
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
            hand_logits, hand_target)
        positives = hand_target > 0
        if positives.any():
            stats["aux_hand"] += float(((hand_logits > 0) & positives).float().sum()
                                       / positives.float().sum()) * n

    deck_logits = aux.opp_deck(context)
    deck_target = batch["opp_deck_hot"]
    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
        deck_logits, deck_target)
    deck_positives = deck_target > 0
    if deck_positives.any():
        stats["aux_deck"] += float(((deck_logits > 0) & deck_positives).float().sum()
                                   / deck_positives.float().sum()) * n

    for head, target_key, stat_key in ((aux.my_use, "my_use_hot", "aux_use"),
                                       (aux.opp_use, "opp_use_hot", "aux_opp_use")):
        use_logits = head(context)
        use_target = batch[target_key]
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
            use_logits, use_target)
        use_positives = use_target > 0
        if use_positives.any():
            stats[stat_key] += float(((use_logits > 0) & use_positives).float().sum()
                                     / use_positives.float().sum()) * n

    counts_predicted = torch.sigmoid(aux.play_counts(context))
    loss = loss + ((counts_predicted - batch["play_counts"]) ** 2).mean()
    stats["aux_counts"] += float((counts_predicted.detach()
                                  - batch["play_counts"]).abs().mean() * 10.0) * n

    board_mask = batch["opp_board_mask"]
    if board_mask.any():
        board_logits = aux.opp_board(context)[board_mask]
        board_target = batch["opp_board_hot"][board_mask]
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
            board_logits, board_target)
        board_positives = board_target > 0
        if board_positives.any():
            stats["aux_board"] += float(((board_logits > 0)
                                         & board_positives).float().sum()
                                        / board_positives.float().sum()) * n

    survives_mask = batch["survives"] >= 0
    if survives_mask.any():
        logits = aux.active_survives(context).squeeze(-1)[survives_mask]
        target = batch["survives"][survives_mask]
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        stats["aux_survives"] += float(((logits > 0).float() == target).float().mean()) * n
    return loss


V21_LOSS_EMA = 0.99          # running-mean decay for the per-head loss normalizers


class AuxHeadsV21(torch.nn.Module):
    """The v2.1 suite (NEXT_MODEL_DESIGN.md section 3): TokenFuture / SideFuture /
    ActionFuture, each a SHARED decoder over the trunk output plus one linear per target.
    TokenFuture reads the per-card token embeddings; the other two read the CLS context.
    Modules are selectable (--v21-modules) so they can be enabled one at a time (spec
    layer 5). Saved as aux_state; the trunk state_dict is untouched, as with the legacy
    head sets, which stay exactly as they were for backward compatibility.

    Per-head loss NORMALIZATION (spec: "to ~unit scale before weighting"): a running mean
    of each component's own raw loss, kept as a buffer (so it survives checkpoints) and
    updated without a host sync. Chosen over fixed hand-derived constants because the raw
    scales here span ~40x AND drift as the model learns (a 1268-way BCE falls fast, a
    9-way CE barely moves), so a constant would only be right at one moment of the run.
    The components are then AVERAGED, not summed, so the TOTAL aux term stays ~1.0 before
    --aux-weight -- i.e. "total aux weight stays 0.1" holds no matter how many heads run."""

    def __init__(self, d_model=128, modules=("token", "side", "action")):
        super().__init__()
        self.enabled = tuple(modules)
        decoder = lambda: torch.nn.Sequential(torch.nn.Linear(d_model, d_model),  # noqa: E731
                                              torch.nn.GELU())
        if "token" in self.enabled:
            self.token_decoder = decoder()
            self.ko_turns = torch.nn.Linear(d_model, NUM_TIMING_CLASSES)
            self.damage = torch.nn.Linear(d_model, DAMAGE_WINDOWS)
            self.token_energy = torch.nn.Linear(d_model, 2)          # +1 / +2
            self.present = torch.nn.Linear(d_model, 1)
        if "side" in self.enabled:
            self.side_decoder = decoder()
            self.prize_turns = torch.nn.Linear(d_model,
                                               2 * V21_PRIZE_K * NUM_TIMING_CLASSES)
            self.deckout = torch.nn.Linear(d_model, 2 * NUM_TIMING_CLASSES)
            self.board = torch.nn.Linear(d_model, V21_BOARD_SLOTS * CARD_VOCAB)
            self.side_energy = torch.nn.Linear(d_model, V21_BOARD_SLOTS)
            self.hand_sizes = torch.nn.Linear(d_model, 2)
            self.opp_hand = torch.nn.Linear(d_model, CARD_VOCAB)
            self.opp_deck = torch.nn.Linear(d_model, CARD_VOCAB)
            self.opp_active = torch.nn.Linear(d_model, CARD_VOCAB)
        if "action" in self.enabled:
            self.action_decoder = decoder()
            self.plays = torch.nn.Linear(d_model, V21_HORIZONS * CARD_VOCAB)
            self.abilities = torch.nn.Linear(d_model, V21_HORIZONS * CARD_VOCAB)
            self.attack = torch.nn.Linear(d_model, V21_HORIZONS * ATTACK_VOCAB)
            self.retreat = torch.nn.Linear(d_model, V21_HORIZONS)
            self.my_use = torch.nn.Linear(d_model, CARD_VOCAB)
            self.opp_use = torch.nn.Linear(d_model, CARD_VOCAB)
        self.register_buffer("loss_scale", torch.ones(len(V21_COMPONENTS)))
        self.register_buffer("loss_seen", torch.zeros(len(V21_COMPONENTS)))

    def normalized(self, name, raw):
        """raw loss -> ~unit scale, dividing by its own running mean (updated in place,
        device-side: no float() and so no host sync in the training step)."""
        index = V21_COMPONENTS.index(name)
        with torch.no_grad():
            value = raw.detach().float()
            if bool(value.numel()):
                update = torch.where(self.loss_seen[index] > 0,
                                     V21_LOSS_EMA * self.loss_scale[index]
                                     + (1.0 - V21_LOSS_EMA) * value, value)
                # a zero loss (fully masked minibatch) must not drag the scale to 0
                keep = value > 0
                self.loss_scale[index] = torch.where(keep, update,
                                                     self.loss_scale[index])
                self.loss_seen[index] = torch.where(keep,
                                                    torch.ones_like(self.loss_seen[index]),
                                                    self.loss_seen[index])
        return raw / self.loss_scale[index].clamp(min=1e-3)


def _v21_masked_ce(logits, labels, stats, key, n):
    """_masked_ce, but `key=None` means "no accuracy stat" (the legacy helper would write a
    None-keyed entry and break the sorted() in the log line)."""
    if key is None:
        mask = labels >= 0
        if not bool(mask.any()):
            return torch.zeros((), device=logits.device)
        return torch.nn.functional.cross_entropy(logits[mask], labels[mask])
    return _masked_ce(logits, labels, stats, key, n)


def _masked_bce(logits, target, mask, stats, key, n):
    """BCE over the rows a mask keeps; `stats[key]` gets recall on the positives."""
    if not bool(mask.any()):
        return torch.zeros((), device=logits.device)
    logits, target = logits[mask], target[mask].float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    positives = target > 0
    if key is not None and bool(positives.any()):
        stats[key] += float(((logits.detach() > 0) & positives).float().sum()
                            / positives.float().sum()) * n
    return loss


def _masked_regression(predicted, target, stats, key, n, scale):
    """Sigmoid regression over the entries with a non-negative target; `stats[key]` gets
    mean absolute error in the label's own units."""
    mask = target >= 0
    if not bool(mask.any()):
        return torch.zeros((), device=predicted.device)
    predicted = torch.sigmoid(predicted)[mask]
    target = target[mask]
    if key is not None:
        stats[key] += float((predicted.detach() - target).abs().mean() * scale) * n
    return ((predicted - target) ** 2).mean()


def aux_losses_v21(aux, context, token_embeddings, batch, stats, n):
    """The v2.1 loss. Every component is normalized to ~unit scale and the enabled ones are
    AVERAGED, so --aux-weight is the weight of the whole suite (see AuxHeadsV21)."""
    terms = []

    def add(name, raw):
        stats[f"aux_{name}"] += float(raw.detach()) * n      # the per-head diagnostic
        terms.append(aux.normalized(name, raw))

    if "token" in aux.enabled:
        width = min(token_embeddings.shape[1], MAX_BOARD_TOKENS)
        tokens = aux.token_decoder(token_embeddings[:, :width])
        add("ko_turns", _v21_masked_ce(aux.ko_turns(tokens).reshape(-1, NUM_TIMING_CLASSES),
                                   batch["v21_ko"][:, :width].reshape(-1),
                                   stats, "aux_ko_acc", n))
        add("damage", _masked_regression(aux.damage(tokens),
                                         batch["v21_damage"][:, :width],
                                         stats, "aux_damage_mae", n, DAMAGE_CAP))
        add("token_energy", _masked_regression(aux.token_energy(tokens),
                                               batch["v21_energy"][:, :width],
                                               stats, None, n, ENERGY_CAP))
        add("present", _masked_regression(aux.present(tokens).squeeze(-1),
                                          batch["v21_present"][:, :width],
                                          stats, None, n, 1.0))
    if "side" in aux.enabled:
        side = aux.side_decoder(context)
        prize_logits = aux.prize_turns(side).view(-1, 2, V21_PRIZE_K, NUM_TIMING_CLASSES)
        prize_target = batch["v21_prize"].view(-1, 2, V21_PRIZE_K)
        for index, name in ((0, "my_prize_turns"), (1, "opp_prize_turns")):
            add(name, _v21_masked_ce(
                prize_logits[:, index].reshape(-1, NUM_TIMING_CLASSES),
                prize_target[:, index].reshape(-1), stats,
                f"aux_{name}_acc" if index == 0 else None, n))
        add("deckout", _v21_masked_ce(
            aux.deckout(side).view(-1, NUM_TIMING_CLASSES),
            batch["v21_deckout"].reshape(-1), stats, None, n))
        add("board", _masked_bce(
            aux.board(side).view(-1, V21_BOARD_SLOTS, CARD_VOCAB),
            batch["v21_board_hot"], batch["v21_board_mask"], stats, None, n))
        add("side_energy", _masked_regression(aux.side_energy(side),
                                              batch["v21_side_energy"], stats,
                                              None, n, SIDE_ENERGY_CAP))
        add("hand_sizes", _masked_regression(aux.hand_sizes(side),
                                             batch["v21_hand_sizes"], stats,
                                             "aux_handsize_mae", n, HAND_SIZE_CAP))
        add("opp_hand", _masked_bce(aux.opp_hand(side), batch["v21_opp_hand_hot"],
                                    batch["v21_opp_hand_mask"], stats,
                                    "aux_opp_hand_recall", n))
        add("opp_deck", _masked_bce(
            aux.opp_deck(side), batch["v21_opp_deck_hot"],
            torch.ones(side.shape[0], dtype=torch.bool, device=side.device),
            stats, None, n))
        add("opp_active", _v21_masked_ce(aux.opp_active(side), batch["v21_opp_active"],
                                     stats, None, n))
    if "action" in aux.enabled:
        action = aux.action_decoder(context)
        mask = batch["v21_action_mask"]
        add("plays", _masked_bce(aux.plays(action).view(-1, V21_HORIZONS, CARD_VOCAB),
                                 batch["v21_play_hot"], mask, stats,
                                 "aux_plays_recall", n))
        add("abilities", _masked_bce(
            aux.abilities(action).view(-1, V21_HORIZONS, CARD_VOCAB),
            batch["v21_ability_hot"], mask, stats, None, n))
        add("attack", _v21_masked_ce(aux.attack(action).reshape(-1, ATTACK_VOCAB),
                                 batch["v21_attack"].reshape(-1), stats,
                                 "aux_attack_acc", n))
        add("retreat", _masked_regression(aux.retreat(action), batch["v21_retreat"],
                                          stats, None, n, 1.0))
        ones = torch.ones(action.shape[0], dtype=torch.bool, device=action.device)
        add("my_use", _masked_bce(aux.my_use(action), batch["v21_my_use_hot"], ones,
                                  stats, None, n))
        add("opp_use", _masked_bce(aux.opp_use(action), batch["v21_opp_use_hot"], ones,
                                   stats, None, n))
    if not terms:
        return torch.zeros((), device=context.device)
    total = torch.stack(terms).mean()
    # the normalizer's own gauge: this should sit near 1.0, so the suite's contribution to
    # the objective is ~--aux-weight no matter how many heads are enabled
    stats["aux_total_normalized"] += float(total.detach()) * n
    return total


class AuxHeadsV22(torch.nn.Module):
    """The v2.2 suite (SEARCH_TRAINING_DESIGN.md, "Aux head rework (v22 suite)"): the v2.1
    layout with `side_energy` cut, `token_energy` replaced by TYPED ATTACHMENTS, the action
    heads rebuilt on select-indexed windows, and four additions (my/opp prize contents,
    stadium at +1/+2, future restriction locks).

    NEW code beside AuxHeadsV21, which is untouched: --heads v21 keeps loading and training
    exactly as before. Saved as aux_state; the trunk state_dict is untouched. The running-
    mean loss normalizer and the AVERAGE over components are verbatim from v2.1, so
    --aux-weight stays the weight of the whole suite."""

    def __init__(self, d_model=128, modules=("token", "side", "action"), suite="v22"):
        super().__init__()
        self.enabled = tuple(modules)
        self.components = V22_COMPONENTS
        # Not a buffer and not in the state_dict: which channels are supervised is a
        # property of the RUN, not of the weights, so a checkpoint stays loadable under any
        # --heads value.
        self.disabled = V22_DISABLED_BY_HEADS.get(suite, V22_DISABLED_BY_HEADS["v22"])
        decoder = lambda: torch.nn.Sequential(torch.nn.Linear(d_model, d_model),  # noqa: E731
                                              torch.nn.GELU())
        if "token" in self.enabled:
            self.token_decoder = decoder()
            self.ko_turns = torch.nn.Linear(d_model, NUM_TIMING_CLASSES)
            self.damage = torch.nn.Linear(d_model, DAMAGE_WINDOWS)
            self.present = torch.nn.Linear(d_model, 1)
            # per (horizon, attachment id): the {0,1,2,3+} count class
            self.typed_attachments = torch.nn.Linear(
                d_model, V22_ATTACH_HORIZONS * ATTACHMENT_VOCAB * V22_COUNT_CLASSES)
        if "side" in self.enabled:
            self.side_decoder = decoder()
            self.prize_turns = torch.nn.Linear(d_model,
                                               2 * V21_PRIZE_K * NUM_TIMING_CLASSES)
            self.deckout = torch.nn.Linear(d_model, 2 * NUM_TIMING_CLASSES)
            self.board = torch.nn.Linear(d_model, V21_BOARD_SLOTS * CARD_VOCAB)
            self.hand_sizes = torch.nn.Linear(d_model, 2)
            self.opp_hand = torch.nn.Linear(d_model, CARD_VOCAB)
            self.opp_deck = torch.nn.Linear(d_model, CARD_VOCAB)
            self.opp_active = torch.nn.Linear(d_model, CARD_VOCAB)
            self.my_prizes = torch.nn.Linear(d_model, CARD_VOCAB)
            self.opp_prizes = torch.nn.Linear(d_model, CARD_VOCAB)
            self.stadium = torch.nn.Linear(d_model, V22_STADIUM_HORIZONS * CARD_VOCAB)
            self.future_locks = torch.nn.Linear(d_model, V22_LOCK_BITS)
        if "action" in self.enabled:
            self.action_decoder = decoder()
            self.plays = torch.nn.Linear(d_model,
                                         V22_WINDOWS * CARD_VOCAB * V22_COUNT_CLASSES)
            self.abilities = torch.nn.Linear(d_model,
                                             V22_WINDOWS * CARD_VOCAB * V22_COUNT_CLASSES)
            self.attack = torch.nn.Linear(d_model, V22_WINDOWS * ATTACK_VOCAB)
            self.retreat = torch.nn.Linear(d_model, V22_WINDOWS)
            self.my_use = torch.nn.Linear(d_model, CARD_VOCAB)
            self.opp_use = torch.nn.Linear(d_model, CARD_VOCAB)
        self.register_buffer("loss_scale", torch.ones(len(V22_COMPONENTS)))
        self.register_buffer("loss_seen", torch.zeros(len(V22_COMPONENTS)))

    def normalized(self, name, raw):
        """raw loss -> ~unit scale, dividing by its own running mean (updated in place,
        device-side: no float() and so no host sync in the training step)."""
        index = self.components.index(name)
        with torch.no_grad():
            value = raw.detach().float()
            if bool(value.numel()):
                update = torch.where(self.loss_seen[index] > 0,
                                     V21_LOSS_EMA * self.loss_scale[index]
                                     + (1.0 - V21_LOSS_EMA) * value, value)
                keep = value > 0          # a fully masked minibatch must not zero the scale
                self.loss_scale[index] = torch.where(keep, update,
                                                     self.loss_scale[index])
                self.loss_seen[index] = torch.where(keep,
                                                    torch.ones_like(self.loss_seen[index]),
                                                    self.loss_seen[index])
        return raw / self.loss_scale[index].clamp(min=1e-3)


def _v22_count_ce(logits, labels, stats, name, n):
    """Count-class CE over a dense card/attachment vocabulary: logits [..., C] against int8
    labels [...] where -1 = masked and 0 = "that card was not played / attached".

    Telemetry is deliberately NOT overall accuracy -- class 0 is ~99.9% of the entries, so
    that number would read 0.999 for any model. `_acc` is accuracy on the entries whose
    LABEL is non-zero (did it get the count right where something actually happened) and
    `_base` is how often that is, i.e. the rate the head has to find."""
    mask = labels >= 0
    if not bool(mask.any()):
        return torch.zeros((), device=logits.device)
    selected = labels[mask].long()
    flat = logits[mask]
    positives = selected > 0
    stats[f"aux_{name}_n"] += n
    stats[f"aux_{name}_base"] += float(positives.float().mean()) * n
    if bool(positives.any()):
        predicted = flat.detach().argmax(dim=-1)
        stats[f"aux_{name}_acc"] += float(
            (predicted[positives] == selected[positives]).float().mean()) * n
        # THE bar this head has to clear. `_base` above is the DENSITY of non-zero cells
        # (0.0128 / 0.0024 / 0.00053 on a grid that is ~99.8% zeros) and reads like a
        # baseline while being nothing of the kind: `_acc` is already scored only on the
        # non-zero cells, so the constant it competes with is "always predict the most
        # common non-zero count", which is 0.83-0.97 (2026-08-06 audit). Against `_base`,
        # typed_attachments/plays/abilities looked like huge wins; against this they are at
        # or below a degenerate predictor.
        counts = torch.bincount(selected[positives], minlength=flat.shape[-1])
        stats[f"aux_{name}_condmaj"] += float(
            counts.max().float() / positives.float().sum()) * n
    return torch.nn.functional.cross_entropy(flat, selected)


def _v22_class_ce(logits, labels, stats, name, n):
    """Plain CE over masked entries: `_acc` = accuracy, `_base` = the frequency of the most
    common class in this minibatch, i.e. what a constant predictor would score."""
    mask = labels >= 0
    if not bool(mask.any()):
        return torch.zeros((), device=logits.device)
    selected = labels[mask]
    flat = logits[mask]
    predicted = flat.detach().argmax(dim=-1)
    stats[f"aux_{name}_n"] += n
    stats[f"aux_{name}_acc"] += float((predicted == selected).float().mean()) * n
    counts = torch.bincount(selected, minlength=flat.shape[-1])
    stats[f"aux_{name}_base"] += float(counts.max().float() / selected.numel()) * n
    return torch.nn.functional.cross_entropy(flat, selected)


def _v22_bce(logits, target, mask, stats, name, n):
    """Masked BCE: `_recall` = recall on the positives, `_base` = the positive rate."""
    if not bool(mask.any()):
        return torch.zeros((), device=logits.device)
    logits, target = logits[mask], target[mask].float()
    positives = target > 0
    stats[f"aux_{name}_n"] += n
    stats[f"aux_{name}_base"] += float(target.mean()) * n
    # Recall alone cannot tell a working head from one that fires on everything, and no
    # precision / F1 / AUC existed anywhere in the trainer. `_fire` is the predicted-positive
    # rate: compare it with `_base`.
    predicted = logits.detach() > 0
    stats[f"aux_{name}_fire"] += float(predicted.float().mean()) * n
    if bool(predicted.any()):
        stats[f"aux_{name}_prec"] += float(
            (predicted & positives).float().sum() / predicted.float().sum()) * n
    if bool(positives.any()):
        stats[f"aux_{name}_recall"] += float(
            (predicted & positives).float().sum() / positives.float().sum()) * n
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)


def _v22_regression(predicted, target, stats, name, n, scale):
    """Sigmoid regression over the non-negative targets: `_mae` in the label's own units,
    `_base` = the mean label (for the 0/1 targets that IS the positive rate), and `_flat`
    = the MAE a CONSTANT predictor would actually score.

    `_flat` exists because `_base` is NOT a baseline and reading it as one inverts the
    verdict (2026-08-06 audit). These heads are trained with MSE, so they chase the
    conditional mean; the constant they have to beat is therefore the label's mean, whose
    MAE is the label's mean absolute deviation -- for a 0/1 target with positive rate p
    that is 2p(1-p), NOT p. On `retreat` (p = 0.120) the two differ by a factor of 1.8:
    `_base` 0.120 is what a degenerate all-ZERO predictor scores, while the honest
    constant-predictor bar is 0.212. Measured MAE 0.157 is between them -- i.e. the head
    IS learning, and AUX_V24_PLAN.md's "worse than predicting the mean" was a misreading of
    this line. `_base` is kept exactly as it was so historical metrics.jsonl rows stay
    comparable; `_flat` is the number to compare `_mae` against."""
    mask = target >= 0
    if not bool(mask.any()):
        return torch.zeros((), device=predicted.device)
    predicted = torch.sigmoid(predicted)[mask]
    target = target[mask]
    stats[f"aux_{name}_n"] += n
    stats[f"aux_{name}_mae"] += float((predicted.detach() - target).abs().mean()
                                      * scale) * n
    stats[f"aux_{name}_base"] += float(target.mean() * scale) * n
    stats[f"aux_{name}_flat"] += float((target - target.mean()).abs().mean() * scale) * n
    return ((predicted - target) ** 2).mean()


def aux_losses_v22(aux, context, token_embeddings, batch, stats, n):
    """The v2.2 loss. Every component is normalized to ~unit scale and the enabled ones are
    AVERAGED, so --aux-weight is the weight of the whole suite -- and EVERY component
    reports an accuracy / recall / MAE next to its label's base rate (no loss-only heads)."""
    terms = []

    def live(name):
        """Is this channel supervised under the run's --heads value? See
        V22_DISABLED_BY_HEADS. A disabled channel must not be COMPUTED either -- the metric
        helpers write their stats as a side effect, so evaluating and dropping one keeps it
        printing a report line while it receives no gradient."""
        return name not in aux.disabled

    def add(name, raw):
        stats[f"aux_{name}"] += float(raw.detach()) * n      # the per-head diagnostic
        terms.append(aux.normalized(name, raw))

    if "token" in aux.enabled:
        width = min(token_embeddings.shape[1], MAX_BOARD_TOKENS)
        tokens = aux.token_decoder(token_embeddings[:, :width])
        add("ko_turns", _v22_class_ce(aux.ko_turns(tokens).reshape(-1, NUM_TIMING_CLASSES),
                                      batch["v21_ko"][:, :width].reshape(-1),
                                      stats, "ko_turns", n))
        add("damage", _v22_regression(aux.damage(tokens),
                                      batch["v21_damage"][:, :width],
                                      stats, "damage", n, DAMAGE_CAP))
        add("present", _v22_regression(aux.present(tokens).squeeze(-1),
                                       batch["v21_present"][:, :width],
                                       stats, "present", n, 1.0))
        if live("typed_attachments"):
            add("typed_attachments", _v22_count_ce(
                aux.typed_attachments(tokens).view(-1, width, V22_ATTACH_HORIZONS,
                                                   ATTACHMENT_VOCAB, V22_COUNT_CLASSES),
                batch["v22_attach"][:, :width], stats, "typed_attachments", n))
    if "side" in aux.enabled:
        side = aux.side_decoder(context)
        prize_logits = aux.prize_turns(side).view(-1, 2, V21_PRIZE_K, NUM_TIMING_CLASSES)
        prize_target = batch["v21_prize"].view(-1, 2, V21_PRIZE_K)
        for index, name in ((0, "my_prize_turns"), (1, "opp_prize_turns")):
            add(name, _v22_class_ce(
                prize_logits[:, index].reshape(-1, NUM_TIMING_CLASSES),
                prize_target[:, index].reshape(-1), stats, name, n))
        if live("deckout"):
            add("deckout", _v22_class_ce(aux.deckout(side), batch["v21_deckout"],
                                         stats, "deckout", n))
        if live("board"):
            add("board", _v22_bce(
                aux.board(side).view(-1, V21_BOARD_SLOTS, CARD_VOCAB),
            batch["v21_board_hot"], batch["v21_board_mask"], stats, "board", n))
        add("hand_sizes", _v22_regression(aux.hand_sizes(side),
                                          batch["v21_hand_sizes"], stats,
                                          "hand_sizes", n, HAND_SIZE_CAP))
        add("opp_hand", _v22_bce(aux.opp_hand(side), batch["v21_opp_hand_hot"],
                                 batch["v21_opp_hand_mask"], stats, "opp_hand", n))
        # SEAT 0 ROWS ONLY. Seat 1's "opponent deck" is the fixed --focus-deck, a game
        # constant the model reads off its own board/hand/deck tokens; training on it taught
        # the trunk to recognise which seat it is (2026-08-06 audit: exactly 24.00 ids/row
        # fired on mover==1, the focus list's distinct-id count, at recall 0.9999). Masking
        # rather than retiring keeps the honest ~half, where the opponent is genuinely drawn
        # from the field.
        # No silent fallback: a data path that drops v22_mover would reintroduce the exact
        # seat-identity leak the mask exists for, unmasked and unannounced.
        assert "v22_mover" in batch, "v22_mover missing -- opp_deck mask would silently drop"
        honest = batch["v22_mover"] == 0
        add("opp_deck", _v22_bce(aux.opp_deck(side), batch["v21_opp_deck_hot"],
                                 honest, stats, "opp_deck", n))
        add("opp_active", _v22_class_ce(aux.opp_active(side), batch["v21_opp_active"],
                                        stats, "opp_active", n))
        if live("my_prizes"):
            add("my_prizes", _v22_bce(aux.my_prizes(side), batch["v22_my_prize_hot"],
                                      batch["v22_my_prize_mask"], stats, "my_prizes", n))
        if live("opp_prizes"):
            add("opp_prizes", _v22_bce(aux.opp_prizes(side), batch["v22_opp_prize_hot"],
                                       batch["v22_opp_prize_mask"], stats,
                                       "opp_prizes", n))
        add("stadium", _v22_class_ce(aux.stadium(side).reshape(-1, CARD_VOCAB),
                                     batch["v22_stadium"].reshape(-1), stats,
                                     "stadium", n))
        add("future_locks", _v22_bce(aux.future_locks(side), batch["v22_locks"],
                                     batch["v22_locks"][:, 0] >= 0, stats,
                                     "future_locks", n))
    if "action" in aux.enabled:
        action = aux.action_decoder(context)
        if live("plays"):
            add("plays", _v22_count_ce(
                aux.plays(action).view(-1, V22_WINDOWS, CARD_VOCAB, V22_COUNT_CLASSES),
                batch["v22_play_counts"], stats, "plays", n))
        add("abilities", _v22_count_ce(
            aux.abilities(action).view(-1, V22_WINDOWS, CARD_VOCAB, V22_COUNT_CLASSES),
            batch["v22_ability_counts"], stats, "abilities", n))
        add("attack", _v22_class_ce(aux.attack(action).reshape(-1, ATTACK_VOCAB),
                                    batch["v22_attack"].reshape(-1), stats, "attack", n))
        add("retreat", _v22_regression(aux.retreat(action), batch["v22_retreat"],
                                       stats, "retreat", n, 1.0))
        ones = torch.ones(action.shape[0], dtype=torch.bool, device=action.device)
        add("my_use", _v22_bce(aux.my_use(action), batch["v21_my_use_hot"], ones,
                               stats, "my_use", n))
        add("opp_use", _v22_bce(aux.opp_use(action), batch["v21_opp_use_hot"], ones,
                                stats, "opp_use", n))
    if not terms:
        return torch.zeros((), device=context.device)
    total = torch.stack(terms).mean()
    # the normalizer's own gauge: this should sit near 1.0, so the suite's contribution to
    # the objective is ~--aux-weight no matter how many heads are enabled
    stats["aux_total_normalized"] += float(total.detach()) * n
    # ...and how many terms that mean was over, so a channel being switched on or off is
    # visible as a step rather than as an unexplained jump in every trend.
    stats["aux_v22_terms"] += len(terms) * n
    return total


class PayabilityHeads(torch.nn.Module):
    """The --attack-aux-weight suite: three binary predictions over the trunk's BOARD token
    embeddings (emission order: my active, my bench, their active, their bench -- so the
    first PAYABLE_BOARD_SLOTS rows are MY in-play Pokemon), PAYABLE_SLOTS columns each
    ([attack 0, attack 1 | ability 0, ability 1] of whatever card sits in that row).

    One head per temporal position: what the engine offers now, at my next MAIN select, and
    anywhere in my next 3. Saved as payability_state -- the trunk state_dict and the
    aux_state of every other head set are untouched, so old checkpoints keep loading."""

    def __init__(self, d_model=128):
        super().__init__()
        self.now = torch.nn.Linear(d_model, PAYABLE_SLOTS)
        self.following = torch.nn.Linear(d_model, PAYABLE_SLOTS)
        self.horizon = torch.nn.Linear(d_model, PAYABLE_SLOTS)


def payability_losses(heads, token_embeddings, batch, stats, n):
    """Masked BCE per head, AVERAGED over the three so --attack-aux-weight is the weight of
    the whole suite. Telemetry is split attack columns (`atk_*`) vs ability columns
    (`abl_*`): masked accuracy (at logit 0) and the label's own positive rate, so "the model
    knows what is payable" is watchable against the base rate it has to beat."""
    width = min(token_embeddings.shape[1], PAYABLE_BOARD_SLOTS)
    rows = token_embeddings[:, :width]
    terms = []
    for name, logits, target in (("now", heads.now(rows), batch["payable_now"]),
                                 ("next", heads.following(rows), batch["payable_next"]),
                                 ("hor", heads.horizon(rows), batch["payable_horizon"])):
        target = target[:, :width]
        mask = target >= 0
        if not bool(mask.any()):
            continue
        terms.append(torch.nn.functional.binary_cross_entropy_with_logits(
            logits[mask], target[mask]))
        for prefix, columns in (("atk", slice(0, ATTACK_SLOTS)),
                                ("abl", slice(ATTACK_SLOTS, PAYABLE_SLOTS))):
            part = mask[:, :, columns]
            if not bool(part.any()):
                continue
            predicted = (logits[:, :, columns].detach() > 0).float()[part]
            actual = target[:, :, columns][part]
            stats[f"{prefix}_{name}_acc"] += float((predicted == actual).float().mean()) * n
            stats[f"{prefix}_{name}_pos"] += float(actual.mean()) * n
    if not terms:
        return torch.zeros((), device=token_embeddings.device)
    total = torch.stack(terms).mean()
    stats["pay_loss"] += float(total.detach()) * n      # the suite's own BCE, before weight
    return total


class PinnedStaging:
    """Reusable page-locked staging for the H2D transfers (--pinned-staging).

    The pre-collated minibatches come from `torch.from_numpy`, i.e. PAGEABLE memory, and a
    `.to(device, non_blocking=True)` out of pageable memory is NOT actually asynchronous --
    the driver has to bounce it through its own staging buffer, so the prefetch stream buys
    nothing. Copying the batch into a page-locked buffer first makes the transfer a real
    async DMA that overlaps the previous minibatch's compute.

    Pinning the whole pre-collated dataset would be the obvious version and is deliberately
    NOT what this does: that is multiple GB of page-locked memory, which on Windows/WDDM
    starves the rest of the machine (the reason the pinning was removed on 07-17). Instead
    a couple of buffer SETS are allocated once, each sized at the largest minibatch seen,
    and every batch is copied into a view of them.

    Buffers are flat 1-D so the per-batch view (`buffer[:numel].view(shape)`) is CONTIGUOUS:
    a strided slice of a 3-D buffer would make `.to()` materialise a pageable temporary and
    undo the whole point. Two sets alternate, each with a CUDA event recorded after its
    transfer, so a set is never rewritten while its DMA is still in flight.

    Pure transport: the bytes handed to the GPU are the bytes `collate` produced.
    """

    def __init__(self, batches, sets=2):
        sizes, dtypes = {}, {}
        for batch in batches:
            for key, value in batch.items():
                sizes[key] = max(sizes.get(key, 0), value.numel())
                dtypes[key] = value.dtype
        self.buffers = [{key: torch.empty(sizes[key], dtype=dtypes[key]).pin_memory()
                         for key in sizes} for _ in range(sets)]
        self.events = [torch.cuda.Event() for _ in range(sets)]
        for event in self.events:
            event.record()                    # so the first reuse-wait is a no-op
        self.slot = 0

    def stage(self, batch):
        """-> (pinned views holding this batch's bytes, the slot's reuse event)."""
        slot = self.slot
        self.slot = (self.slot + 1) % len(self.buffers)
        self.events[slot].synchronize()       # last DMA out of this slot has landed
        buffers = self.buffers[slot]
        staged = {}
        for key, value in batch.items():
            buffer = buffers.get(key)
            if buffer is None or buffer.dtype != value.dtype:
                staged[key] = value           # key absent at construction: ship as-is
                continue
            view = buffer[:value.numel()].view(value.shape)
            view.copy_(value)
            staged[key] = view
        return staged, self.events[slot]


class CollateFeeder:
    """Collate minibatch k+1 on a worker THREAD while the caller trains minibatch k.

    Collating is numpy/torch work that spends nearly all of its time inside routines that
    release the GIL, so overlapping it with the GPU step is a real win and not a
    context-switch tax. Iterating the feeder yields the batches in order and appends each
    one to `collected`, so later epochs reuse the same objects (they are collated ONCE, as
    before). An exception in the thread is re-raised in the consumer, which is what keeps
    the label invariant checks fatal.

    Pure scheduling: the batches are the bytes `build` produced, in the same order."""

    def __init__(self, groups, build, collected, depth=2):
        self.queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.collected = collected
        self.thread = threading.Thread(target=self._run, args=(groups, build), daemon=True)
        self.thread.start()

    def _run(self, groups, build):
        try:
            for group in groups:
                if self.stop.is_set():
                    break
                self.queue.put(("batch", build(group)))
        except BaseException as error:                      # noqa: BLE001 - re-raised below
            self.queue.put(("error", error))
        self.queue.put(("done", None))

    def __iter__(self):
        while True:
            kind, payload = self.queue.get()
            if kind == "error":
                raise payload
            if kind == "done":
                return
            self.collected.append(payload)
            yield payload

    def close(self):
        """Let the thread out of a blocked put when the consumer stopped early (--kl-stop),
        so a run that brakes often does not accumulate parked threads."""
        self.stop.set()
        while self.thread.is_alive():
            try:
                self.queue.get(timeout=0.05)
            except queue.Empty:
                pass
        self.thread.join(timeout=1.0)


class BatchPrefetcher:
    """One-batch-ahead H2D prefetch on a dedicated copy stream (pinned tensors +
    non_blocking): the next minibatch's transfer overlaps the current one's compute."""

    def __init__(self, batches, device, staging=None):
        self.batches = iter(batches)
        self.device = device
        self.staging = staging
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self._preload()

    def _preload(self):
        batch = next(self.batches, None)
        if batch is None:
            self.next_batch = None
            return
        event = None
        if self.staging is not None:
            batch, event = self.staging.stage(batch)
        with torch.cuda.stream(self.stream):
            self.next_batch = {key: value.to(self.device, non_blocking=True)
                               for key, value in batch.items()}
            if event is not None:
                event.record(self.stream)     # frees the staging slot once the DMA lands

    def __iter__(self):
        while self.next_batch is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = self.next_batch
            for value in batch.values():
                value.record_stream(torch.cuda.current_stream())
            self._preload()
            yield batch


def ppo_update(model, optimizer, flat, args, device, aux=None, payability=None,
               anchor=None, aux_v23=None):
    advantages = np.array([d["advantage"] for d in flat], dtype=np.float32)
    # Std floor 0.1: safety only -- measured std sits ~0.40 in practice (2026-07-26
    # instrumentation), so it never binds today; kept so a future near-converged critic
    # can't re-amplify noise. The age-locked kl blowups were PIPELINE STALENESS (batches
    # arriving 1 update old under --overlap; kl_first tracked 60-70% of total kl through
    # the takeoff) -- fixed by launching with --no-overlap, not by this floor.
    raw_advantage_std = float(advantages.std())
    scale = max(raw_advantage_std, 0.1)
    center = float(advantages.mean())
    for d in flat:
        d["advantage"] = (d["advantage"] - center) / scale

    stats = Counter()
    stats["adv_std_raw"] = raw_advantage_std     # diagnostic: does the floor ever bind?
    rng = random.Random(0)
    model.train()
    # Length-bucketed pre-collate: sorting by token count makes near-uniform batches
    # (fewer padding tokens through the transformer) and collating ONCE across epochs
    # halves the Python collate overhead; epochs reshuffle the batch ORDER.
    collate_start = time.time()
    order = sorted(flat, key=lambda d: dense_shape(d["tokens"])[0])
    groups = [order[start:start + args.minibatch]
              for start in range(0, len(order), args.minibatch)]
    checks = aux is not None and getattr(args, "v21_checks", True) \
        and args.heads in ("v21", "v22", "v23", "v24", "v25", "v26")

    collate_seconds = [time.time() - collate_start]     # the bucketing above, then the thread

    def build_batch(group):
        start = time.time()
        batch = collate(group, heads=args.heads, aux_labels=aux is not None,
                        payability=payability is not None)
        if checks:
            # Spec layer 2, once per collated minibatch (they are built once and reused
            # across epochs) and on the CPU side, so the invariants cost no GPU sync.
            (v22_batch_checks if args.heads in ("v22", "v23", "v24", "v25", "v26")
             else v21_batch_checks)(batch)
            if args.heads in ("v23", "v24", "v25", "v26"):
                aux_head_labels.batch_checks(batch)
        collate_seconds[0] += time.time() - start
        return batch

    prebuilt = []
    gpu_start = time.time()
    # The batches themselves are NOT pinned: page-locking ~10+ GB of them starved the
    # machine (07-17). --pinned-staging instead keeps two REUSABLE pinned buffer sets
    # (tens of MB, sized at the largest minibatch) and copies each batch through them, so
    # the prefetch stream's non_blocking H2D is a real overlapping DMA. Byte-neutral.
    staging = None
    feeder = None
    for epoch in range(args.epochs):
        if epoch == 0:
            # First pass: the batches do not exist yet, so they are COLLATED ON A THREAD one
            # ahead of the GPU instead of all up front (the whole collate phase used to sit
            # on the serial critical path). Staging buffers need the finished set to size
            # themselves, so they join from the second epoch on; staging is pure transport.
            feeder = CollateFeeder(groups, build_batch, prebuilt)
            source = feeder
        else:
            if staging is None and device == "cuda" and args.pinned_staging:
                staging = PinnedStaging(prebuilt)
            rng.shuffle(prebuilt)
            source = prebuilt
        epoch_batches = BatchPrefetcher(source, device, staging=staging) \
            if device == "cuda" \
            else ({key: value.to(device) for key, value in batch.items()}
                  for batch in source)
        for batch in epoch_batches:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=bool(args.bf16 and device == "cuda")):
                inputs = (batch["tokens"].float(), batch["owners"], batch["zones"],
                          batch["padding"], batch["globals"],
                          batch["options"].float(), batch["option_mask"])
                identity = {"card_ids": batch["card_ids"]} if "card_ids" in batch else {}
                token_embeddings = None
                if payability is not None \
                        or (aux is not None and args.heads in ("full", "v21", "v22", "v23", "v24", "v25", "v26")):
                    logits, value, context, token_embeddings = \
                        model.policy_value_tokens(*inputs, **identity)
                elif aux is not None:
                    logits, value, context = model.policy_value_context(*inputs, **identity)
                else:
                    logits, value = model.policy_value(*inputs, **identity)
                logits = logits.float()
                value = value.float()
                log_probabilities = torch.log_softmax(logits, dim=-1)
                chosen_logprob = log_probabilities.gather(
                    1, batch["chosen"].unsqueeze(1)).squeeze(1)
                ratio = torch.exp(chosen_logprob - batch["logprob"])
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                policy_loss = -torch.min(ratio * batch["advantage"],
                                         clipped * batch["advantage"]).mean()
                # --value-outcome-weight: blends the value target between the GAE return
                # (bootstrapped, distance-attenuated) and the FINAL game outcome (full-
                # horizon, AlphaZero-style, noisier). 0 = original behavior. Advantages
                # stay GAE regardless -- only the value head's teacher changes.
                w = args.value_outcome_weight
                value_target = ((1.0 - w) * batch["return"]
                                + w * batch["outcome"]) if w > 0 else batch["return"]
                value_loss = ((value - value_target) ** 2).mean()
                probabilities = torch.softmax(logits, dim=-1)
                entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                # --ppo-anchor-weight: KL(current || frozen loaded policy) on every
                # decision. The consolidation-phase leash of interleaved PPO/EI:
                # reward gradients sharpen the policy while the anchor defends the
                # rare search-taught behaviors PPO alone erodes. anchor is None
                # unless the flag is set, so plain PPO runs never enter this block.
                if anchor is not None and args.ppo_anchor_weight > 0:
                    with torch.no_grad():
                        anchor_logits, _anchor_value = anchor.policy_value(*inputs,
                                                                           **identity)
                        anchor_log = torch.log_softmax(anchor_logits.float(), dim=-1)
                    anchor_kl = (probabilities * (log_probabilities - anchor_log)) \
                        .sum(dim=-1).mean()
                    loss = loss + args.ppo_anchor_weight * anchor_kl
                if aux is not None and args.heads in ("v22", "v23", "v24", "v25", "v26"):
                    loss = loss + args.aux_weight * aux_losses_v22(
                        aux, context.float(),
                        token_embeddings.float(), batch, stats, len(batch["chosen"]))
                    if aux_v23 is not None:
                        # v23 is added as its OWN averaged suite next to v22's, at the same
                        # --aux-weight. Its option heads read exactly what policy_score
                        # reads: cat([board context, option features]).
                        loss = loss + args.aux_weight * aux_head_labels.aux_losses_v23(
                            aux_v23, context.float(), token_embeddings.float(),
                            inputs[5], batch, stats, len(batch["chosen"]))
                elif aux is not None and args.heads == "v21":
                    loss = loss + args.aux_weight * aux_losses_v21(
                        aux, context.float(),
                        token_embeddings.float(), batch, stats, len(batch["chosen"]))
                elif aux is not None and args.heads == "full":
                    loss = loss + args.aux_weight * aux_losses_full(
                        aux, context.float(),
                        token_embeddings.float(), batch, stats, len(batch["chosen"]))
                elif aux is not None:
                    loss = loss + args.aux_weight * aux_losses(
                        aux, context.float(), batch, stats, len(batch["chosen"]))
                if payability is not None:
                    loss = loss + args.attack_aux_weight * payability_losses(
                        payability, token_embeddings.float(), batch, stats,
                        len(batch["chosen"]))

            optimizer.zero_grad()
            loss.backward()
            trained_parameters = list(model.parameters()) \
                + (list(aux.parameters()) if aux is not None else []) \
                + (list(aux_v23.parameters()) if aux_v23 is not None else []) \
                + (list(payability.parameters()) if payability is not None else [])
            grad_norm = torch.nn.utils.clip_grad_norm_(trained_parameters,
                                                       args.max_grad_norm)
            if torch.isfinite(grad_norm):
                optimizer.step()
            n = len(batch["chosen"])
            stats["n"] += n
            # 2026-08-11 audit: the aux suite shares this clip with a policy gradient
            # orders of magnitude smaller -- log the joint norm so clip saturation
            # (grad_norm >> max_grad_norm) is observable instead of hypothetical.
            stats["grad_norm"] += float(grad_norm) * n
            stats["grad_norm_max"] = max(stats["grad_norm_max"], float(grad_norm))
            stats["policy_loss"] += float(policy_loss.detach()) * n
            stats["value_loss"] += float(value_loss.detach()) * n
            stats["entropy"] += float(entropy.detach()) * n
            stats["kl"] += float((batch["logprob"]
                                  - chosen_logprob.detach()).mean()) * n
            if anchor is not None and args.ppo_anchor_weight > 0:
                stats["anchor_kl"] += float(anchor_kl.detach()) * n
            if "kl_first" not in stats:
                # diagnostic: kl of the very first minibatch, BEFORE this update moved
                # anything. ~0 = data matches the policy; large = the batch was stale
                # on arrival (pipeline lag / publish mismatch), i.e. the movement being
                # blamed on this update actually happened before it.
                stats["kl_first"] = float((batch["logprob"]
                                           - chosen_logprob.detach()).mean())
                # ...and the same quantity from an eval-mode fp32 forward (matching the
                # generation path). kl_first >> kl_first_eval = the gap is AUTOCAST
                # measurement noise, not real weight mismatch: bf16 logit error on
                # sampled actions reads as positive kl and grows with logit magnitude.
                with torch.no_grad():
                    model.eval()
                    eval_logits, _ = model.policy_value(*inputs, **identity)
                    model.train()
                    eval_logprob = torch.log_softmax(
                        eval_logits.float(), dim=-1).gather(
                        1, batch["chosen"].unsqueeze(1)).squeeze(1)
                    stats["kl_first_eval"] = float(
                        (batch["logprob"] - eval_logprob).mean())
            stats["clipfrac"] += float((torch.abs(ratio.detach() - 1.0)
                                        > args.clip).float().mean()) * n
            # KL brake: once this update has already moved the policy past the target,
            # stop consuming the remaining minibatches/epochs. Without it a big update
            # feeds the next one (ratios explode, clip saturates ~0.95, value EV dies)
            # -- the runaway that killed the d448 focus run and a384 at iter ~357.
            if args.kl_stop and stats["kl"] / stats["n"] > args.kl_stop:
                stats["kl_stopped"] = 1.0
                break
        else:
            continue
        break
    model.eval()
    if feeder is not None:
        feeder.close()                # --kl-stop can leave it mid-queue
    stats["collate_seconds"] = collate_seconds[0]
    if device == "cuda":
        if staging is not None:
            # The pinned buffers are about to be freed; no DMA may still be reading them
            # (--kl-stop can break out with a prefetched transfer still in flight).
            torch.cuda.synchronize()
            staging = None
        torch.cuda.empty_cache()      # bucketed shapes fragment the caching allocator
    stats["gpu_seconds"] = time.time() - gpu_start
    n = max(1, stats["n"])
    timing_keys = {"collate_seconds", "gpu_seconds", "kl_stopped",
                   "adv_std_raw", "kl_first", "kl_first_eval", "grad_norm_max"}
    return {key: (stats[key] if key in timing_keys else stats[key] / n)
            for key in stats if key != "n"}


# --- Expert-iteration update (--ei). The PPO objective is REPLACED, not extended: no
# ratio, no clip, no advantage. `ppo_update` above is untouched so every other run is
# byte-identical. ------------------------------------------------------------------- #
_RICH_TOKEN_SLICE = (TOKEN_FEATURE_DIM_FULL_RICH - RICH_CARD_DIM,
                     TOKEN_FEATURE_DIM_FULL_RICH)
_RICH_GLOBAL_SLICE = (GLOBAL_FEATURE_DIM_FULL + SELECT_CONTEXT_DIM,
                      GLOBAL_FEATURE_DIM_FULL + SELECT_CONTEXT_DIM + RICH_GLOBAL_DIM)


def _apply_rich_dropout(batch, fraction, generator=None):
    """Zero the ENGINE-EFFECT (rich) block on a random `fraction` of TRAINING rows.

    Search leaves DO get live rich features now (DumpSearchState, addendum 13), so this is
    no longer a necessity -- it is the light robustness measure of design item 8: the value
    head stays calibrated with and without the block, which matters because a bundle
    running on the official binary has no DumpState at all. Rows, not leaves: the leaves
    are always real."""
    if fraction <= 0.0:
        return 0
    rows = batch["tokens"].shape[0]
    mask = torch.rand(rows, device=batch["tokens"].device, generator=generator) < fraction
    if not bool(mask.any()):
        return 0
    batch["tokens"][mask, :, _RICH_TOKEN_SLICE[0]:_RICH_TOKEN_SLICE[1]] = 0
    batch["globals"][mask, _RICH_GLOBAL_SLICE[0]:_RICH_GLOBAL_SLICE[1]] = 0
    return int(mask.sum())


def ei_update(model, optimizer, flat, args, device, aux=None, payability=None,
              anchor=None, aux_v23=None):
    """One expert-iteration update.

    policy  CE(policy logits, PRUNED VISIT TARGET) on SEARCHED decisions only
            + --anchor-weight * KL(current || frozen warm-start policy) on ALL decisions
    value   MSE against the game OUTCOME on all decisions, optionally blended with the
            SEARCH ROOT VALUE on searched ones (--search-value-weight, addendum 11)
    aux     payability (--attack-aux-weight) and the v22 suite (--aux-weight), unchanged
    """
    stats = Counter()
    rng = random.Random(0)
    model.train()
    collate_start = time.time()
    order = sorted(flat, key=lambda d: dense_shape(d["tokens"])[0])
    groups = [order[start:start + args.minibatch]
              for start in range(0, len(order), args.minibatch)]
    checks = aux is not None and getattr(args, "v21_checks", True) \
        and args.heads in ("v21", "v22", "v23", "v24", "v25", "v26")
    collate_seconds = [time.time() - collate_start]

    def build_batch(group):
        start = time.time()
        batch = collate(group, heads=args.heads, aux_labels=aux is not None,
                        payability=payability is not None, search=True)
        if checks:
            (v22_batch_checks if args.heads in ("v22", "v23", "v24", "v25", "v26")
             else v21_batch_checks)(batch)
            if args.heads in ("v23", "v24", "v25", "v26"):
                aux_head_labels.batch_checks(batch)
        collate_seconds[0] += time.time() - start
        return batch

    prebuilt = []
    gpu_start = time.time()
    staging = None
    feeder = None
    value_weight = float(getattr(args, "search_value_weight", 0.0))
    for epoch in range(args.epochs):
        if epoch == 0:
            feeder = CollateFeeder(groups, build_batch, prebuilt)
            source = feeder
        else:
            if staging is None and device == "cuda" and args.pinned_staging:
                staging = PinnedStaging(prebuilt)
            rng.shuffle(prebuilt)
            source = prebuilt
        epoch_batches = BatchPrefetcher(source, device, staging=staging) \
            if device == "cuda" \
            else ({key: value.to(device) for key, value in batch.items()}
                  for batch in source)
        for batch in epoch_batches:
            stats["rich_dropped"] += _apply_rich_dropout(batch, args.rich_dropout)
            inputs = (batch["tokens"].float(), batch["owners"], batch["zones"],
                      batch["padding"], batch["globals"],
                      batch["options"].float(), batch["option_mask"])
            identity = {"card_ids": batch["card_ids"]} if "card_ids" in batch else {}
            token_embeddings = None
            if payability is not None \
                    or (aux is not None and args.heads in ("full", "v21", "v22", "v23", "v24", "v25", "v26")):
                logits, value, context, token_embeddings = \
                    model.policy_value_tokens(*inputs, **identity)
            elif aux is not None:
                logits, value, context = model.policy_value_context(*inputs, **identity)
            else:
                logits, value = model.policy_value(*inputs, **identity)
            logits = logits.float()
            value = value.float()
            log_probabilities = torch.log_softmax(logits, dim=-1)
            searched = batch["searched"]
            count = int(searched.sum())
            if count:
                target = batch["search_target"]
                cross_entropy = -(target * log_probabilities).sum(dim=-1)
                policy_loss = (cross_entropy * searched.float()).sum() / count
            else:
                policy_loss = logits.sum() * 0.0
            # KL(current || anchor) on EVERY decision: the BC-style leash that keeps the
            # distilled policy from drifting off the warm start on the 75% of decisions
            # search never looked at.
            anchor_kl = logits.sum() * 0.0
            if anchor is not None and args.anchor_weight > 0:
                with torch.no_grad():
                    anchor_logits, _anchor_value = anchor.policy_value(*inputs, **identity)
                    anchor_log = torch.log_softmax(anchor_logits.float(), dim=-1)
                probabilities = log_probabilities.exp()
                anchor_kl = (probabilities * (log_probabilities - anchor_log)).sum(
                    dim=-1).mean()
            value_target = batch["outcome"]
            if value_weight > 0:
                blended = ((1.0 - value_weight) * batch["outcome"]
                           + value_weight * batch["search_value"])
                value_target = torch.where(searched, blended, batch["outcome"])
            value_loss = ((value - value_target) ** 2).mean()
            entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1).mean()
            loss = (policy_loss + args.anchor_weight * anchor_kl
                    + args.vf_coef * value_loss)
            if aux is not None and args.heads in ("v22", "v23", "v24", "v25", "v26"):
                loss = loss + args.aux_weight * aux_losses_v22(
                    aux, context.float(), token_embeddings.float(), batch, stats,
                    len(batch["chosen"]))
                if aux_v23 is not None:
                    # see ppo_update: same suite, same weight, same option-row input
                    loss = loss + args.aux_weight * aux_head_labels.aux_losses_v23(
                        aux_v23, context.float(), token_embeddings.float(),
                        inputs[5], batch, stats, len(batch["chosen"]))
            elif aux is not None and args.heads == "v21":
                loss = loss + args.aux_weight * aux_losses_v21(
                    aux, context.float(), token_embeddings.float(), batch, stats,
                    len(batch["chosen"]))
            elif aux is not None:
                loss = loss + args.aux_weight * aux_losses(
                    aux, context.float(), batch, stats, len(batch["chosen"]))
            if payability is not None:
                loss = loss + args.attack_aux_weight * payability_losses(
                    payability, token_embeddings.float(), batch, stats,
                    len(batch["chosen"]))

            optimizer.zero_grad()
            loss.backward()
            trained_parameters = list(model.parameters()) \
                + (list(aux.parameters()) if aux is not None else []) \
                + (list(aux_v23.parameters()) if aux_v23 is not None else []) \
                + (list(payability.parameters()) if payability is not None else [])
            grad_norm = torch.nn.utils.clip_grad_norm_(trained_parameters,
                                                       args.max_grad_norm)
            if torch.isfinite(grad_norm):
                optimizer.step()
            n = len(batch["chosen"])
            stats["n"] += n
            stats["policy_loss"] += float(policy_loss.detach()) * n
            stats["value_loss"] += float(value_loss.detach()) * n
            stats["entropy"] += float(entropy.detach()) * n
            stats["anchor_kl"] += float(anchor_kl.detach()) * n
            stats["searched_rows"] += count
            with torch.no_grad():
                # The falsified-blend watch (addendum 11): EV of the critic against the
                # OUTCOME, split by whether the row's value target was blended. If the
                # blend is hedging rather than denoising, the searched block's EV(out)
                # falls away from the unsearched block's -- kill the flag when it does.
                residual = (value - batch["outcome"]) ** 2
                for name, mask in (("searched", searched), ("unsearched", ~searched)):
                    rows = int(mask.sum())
                    if not rows:
                        continue
                    outcomes = batch["outcome"][mask]
                    variance = float(outcomes.var(unbiased=False))
                    stats[f"ev_{name}_n"] += rows
                    if variance > 1e-9:
                        stats[f"ev_{name}"] += rows * (
                            1.0 - float(residual[mask].mean()) / variance)
    model.eval()
    if feeder is not None:
        feeder.close()
    stats["collate_seconds"] = collate_seconds[0]
    if device == "cuda":
        if staging is not None:
            torch.cuda.synchronize()
            staging = None
        torch.cuda.empty_cache()
    stats["gpu_seconds"] = time.time() - gpu_start
    n = max(1, stats["n"])
    out = {}
    for key in stats:
        if key in ("n", "ev_searched", "ev_unsearched", "ev_searched_n",
                   "ev_unsearched_n"):
            continue
        out[key] = stats[key] if key in ("collate_seconds", "gpu_seconds",
                                         "searched_rows", "rich_dropped") \
            else stats[key] / n
    for name in ("searched", "unsearched"):
        rows = stats.get(f"ev_{name}_n", 0)
        out[f"ev_out_{name}"] = (stats[f"ev_{name}"] / rows) if rows else float("nan")
    out["searched_fraction"] = stats["searched_rows"] / n
    return out


def explained_variance(predictions, targets):
    variance = float(np.var(targets))
    if variance < 1e-9:
        return float("nan")
    return 1.0 - float(np.var(targets - predictions)) / variance


def _require_v23_engine():
    """--heads v23 REFUSES to start on an engine without the DumpState attackCost /
    energyOrder fields (engine_src/build_v23). Those fields carry L1's cost evidence and all
    of T2; on an older DLL both would simply mask out and the two headline heads would train
    on nothing at all -- silently, for a whole run. So probe one real dump here and abort.

    The probe plays one throwaway seeded battle to first blood-free observation, decodes its
    state blob exactly the way the generator does, and looks for the fields on any in-play
    Pokemon that has an attack."""
    from src.game.encode_rich import rich_state
    dll = os.environ.get("CG_DLL", "<default>")
    decks = load_corpus_decks()
    battle = _Battle(list(decks[0]), list(decks[0]), 1)
    try:
        observation, moves = battle.observation, 0
        while observation["current"]["result"] == -1 and moves < 40:
            cards, _players = rich_state(observation)
            if cards and any("attackCost" in card for card in cards.values()):
                print(f"v23 engine check: attackCost/energyOrder present in {dll}",
                      flush=True)
                return
            select = observation.get("select")
            observation = battle.select([] if select is None else [0])
            moves += 1
    finally:
        battle.finish()
    raise SystemExit(
        "--heads v23 needs the v23 engine build: no `attackCost` field in any DumpState "
        f"decode from CG_DLL={dll}.\n"
        "  Set CG_DLL=engine_src/build_v23/cg.dll (the launch kits already prefer it).\n"
        "  Refusing to start: on an older DLL the attachment-need cost evidence and the "
        "whole effective_cost_delta head would mask out silently.")


def save_checkpoint(path, model, optimizer, iteration, args, aux=None, payability=None,
                    aux_v23=None):
    torch.save({
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "token_feature_dim": TOKEN_DIM, "global_feature_dim": GLOBAL_DIM,
        "option_feature_dim": OPTION_DIM, "num_zones": ZONE_COUNT,
        "d_model": model.config.d_model, "num_layers": model.config.num_layers,
        "num_heads": model.config.num_heads,
        "feedforward_dim": model.config.feedforward_dim,
        "solver_features": True,
        # v2 loaders need these to rebuild the trunk and to know which encoder to run;
        # absent/0 in every v1 checkpoint, which is exactly how the old guides read them.
        "encoding": ENCODING,
        "card_vocab": model.config.card_vocab,
        "card_embedding_dim": model.config.card_embedding_dim,
        "aux_state": ({key: value.cpu() for key, value in aux.state_dict().items()}
                      if aux is not None else None),
        # Absent/None in every checkpoint written before --attack-aux-weight existed, which
        # is exactly how the resume path reads it (start the heads fresh).
        "payability_state": ({key: value.cpu() for key, value
                              in payability.state_dict().items()}
                             if payability is not None else None),
        # Its OWN key: absent/None in every pre-v23 checkpoint, which is exactly how
        # the resume path reads it (start the v23 heads fresh, nothing else touched).
        "aux_v23_state": ({key: value.cpu() for key, value
                           in aux_v23.state_dict().items()}
                          if aux_v23 is not None else None),
        "optimizer": optimizer.state_dict(), "iteration": iteration,
        "config": {key: value for key, value in vars(args).items()
                   if isinstance(value, (int, float, str, bool, type(None)))},
    }, path)


def run_probe(pool, version, args, seed_base):
    from eval_panel import OPPONENTS
    opponents = list(OPPONENTS)
    if args.opponent_bundle or args.probe_bundle:
        # Probe (GREEDY) vs the loaded bundle -- the in-training version of the manual
        # snapshot-vs-specialist eval.
        opponents.append("specialist")
    per_opponent = max(1, args.probe_games // len(opponents))
    tasks = [("probe", seed_base + i, version, opponents[i % len(opponents)])
             for i in range(per_opponent * len(opponents))]
    wins = {}
    behavior = Counter()
    dark = Counter()
    games = 0
    for record in pool.imap_unordered(run_task, tasks):
        wins.setdefault(record["opponent"], []).append(record["win"])
        counters = record.get("behavior") or {}
        games += 1
        for key in ("abilities", "attachments", "surplus_attachments"):
            behavior[key] += counters.get(key, 0)
        dark.update(counters.get("dark_targets") or {})
    rates = {name: float(np.mean(values)) for name, values in sorted(wins.items())}
    rates["POOLED"] = float(np.mean([w for values in wins.values() for w in values]))
    # Behavioural counters (design doc): plain per-game aggregates from finished games.
    # Instrumentation, not features -- nothing reads them back into a decision.
    if games:
        rates["_behavior"] = {key: value / games for key, value in behavior.items()}
        rates["_behavior"]["dark_top"] = [
            [int(card), count / games] for card, count in dark.most_common(3)]
    return rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="run1")
    parser.add_argument("--iterations", type=int, default=1000000)
    parser.add_argument("--games-per-iter", type=int, default=512)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 8) - 2))
    parser.add_argument("--lr", type=float, default=2.5e-4)
    # Linear LR anneal (owner 2026-08-10, d128_mask 700->1000 polish experiment): between
    # --lr-anneal-start and --lr-anneal-end the lr interpolates from --lr down to
    # --lr-final; flat at --lr before, flat at --lr-final after. Recomputed from the
    # iteration number every iteration and written into every optimizer param group, so it
    # is resume-safe AND overrides the lr that optimizer.load_state_dict restores from the
    # checkpoint (which otherwise silently wins over a changed --lr). 0 0 = off (default,
    # existing runs bit-identical).
    parser.add_argument("--lr-anneal-start", type=int, default=0)
    parser.add_argument("--lr-anneal-end", type=int, default=0,
                        help="anneal lr linearly from --lr at --lr-anneal-start to "
                             "--lr-final at this iteration; 0 disables")
    parser.add_argument("--lr-final", type=float, default=2.5e-5)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--kl-stop", type=float, default=1.0,
                        help="stop the update early once its running mean kl exceeds "
                             "this (0 disables); divergence brake, see iter-357 incident")
    parser.add_argument("--tf32", action="store_true",
                        help="TF32 matmuls in the training step. Default OFF after the "
                             "2026-07-28 live test: ~20-25%% train speedup but "
                             "kl_first_eval rose 0.002->0.03 (precision drift EXCEEDING "
                             "real per-update movement -- the bf16 mechanism class at "
                             "1/50 amplitude). Not worth mushing the guard telemetry.")
    parser.add_argument("--value-outcome-weight", type=float, default=0.0,
                        help="blend of the value-head target: 0 = pure GAE return "
                             "(default, original behavior), 1 = pure final game outcome "
                             "(full-horizon, AlphaZero-style), between = mix. Advantages "
                             "stay GAE regardless.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=256)
    # Opt-in action rules (owner 2026-08-09), comma-separated; "" = all off (the default
    # keeps every existing run/checkpoint bit-identical). Known: counter_cap (damage
    # counters may not be placed onto an already-dead target while a live one is offered;
    # state_encoder.counter_option_mask), counter_shield (owner 2026-08-12: no counters onto
    # an opponent Pokemon shielded by Mist Energy / Rock Fighting Energy;
    # state_encoder.shielded_counter_mask). Propagated to workers via CG_ACTION_RULES.
    # require_play_supporter_from_meowth_ex, require_play_fetched_card and
    # require_evolve_before_shuffle_draw are accepted too, but their LINE-RULE halves
    # only run in the bundles' search (this rollout loop applies mask surfaces only), so
    # enabling them here changes menus at most.
    parser.add_argument("--action-rules", default="",
                        help="comma-separated opt-in action rules, e.g. counter_cap")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    # Linear entropy-coefficient anneal (owner 2026-08-10), same shape and semantics as
    # the lr anneal above: between --ent-anneal-start and --ent-anneal-end the coefficient
    # interpolates from --ent-coef down to --ent-final, flat outside the window, stateless
    # in the iteration number (resume-safe). Anneal to a NONZERO floor: ent-coef 0 risks
    # entropy collapse (deterministic self-play -> no data diversity, exploitable policy);
    # the watchdogs' ent<0.15 pathological kill is the backstop. 0 0 = off (default).
    parser.add_argument("--ent-anneal-start", type=int, default=0)
    parser.add_argument("--ent-anneal-end", type=int, default=0,
                        help="anneal ent-coef linearly from --ent-coef at "
                             "--ent-anneal-start to --ent-final at this iteration; "
                             "0 disables")
    parser.add_argument("--ent-final", type=float, default=2e-3)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--aux-weight", type=float, default=0.0,
                        help="per-component aux-head loss weight; 0 = bare actor-critic "
                             "(run1 semantics), >0 trains the --heads set on the trunk")
    parser.add_argument("--heads",
                        choices=("family1", "full", "v21", "v22", "v23", "v24", "v25", "v26"),
                        default="v26",
                        help="which aux-head set --aux-weight trains: family1 = the "
                             "grounded-validated 4; full = the MY_MODEL_DESIGN set; "
                             "v21 = the NEXT_MODEL_DESIGN section-3 suite (TokenFuture / "
                             "SideFuture / ActionFuture, per-head loss normalization); "
                             "v22 = the SEARCH_TRAINING_DESIGN rework of that suite "
                             "(typed attachments, select-indexed action windows, prize "
                             "contents, stadium, future restriction locks); "
                             "v23 = v22 ++ the AUX_V23_DESIGN option-pathway groups "
                             "(attachment need, placement conversion, bench contribution); "
                             "v24 = the same modules with the post-audit (2026-08-06) "
                             "supervised set -- my_prizes off as a label leak, "
                             "prizes_donated back on now that the KO test is fixed. "
                             "v23 and v24 both need the build_v23 engine")
    parser.add_argument("--attack-aux-weight", type=float, default=0.0,
                        help="weight of the PAYABILITY head suite (0 = off, the default, "
                             "and nothing about the run changes): three masked-BCE heads "
                             "over the board tokens predicting which of a Pokemon's attacks "
                             "AND abilities the ENGINE offers now / at my next MAIN select "
                             "/ anywhere in my next 3 MAIN selects. Dense signal for 'which "
                             "energies power up which attacks and abilities'; independent "
                             "of --heads and of --aux-weight")
    parser.add_argument("--v21-modules", default="token,side,action",
                        help="--heads v21 only: comma list of the modules to train "
                             "(token, side, action). Staged enablement, spec layer 5: "
                             "bring them up one at a time so a metric wobble implicates "
                             "the last one enabled")
    parser.add_argument("--no-label-invariant-checks", dest="v21_checks", action="store_false",
                        help="turn OFF the label invariant checks (default ON): "
                             "per-minibatch invariants plus sampled assemble-time "
                             "cross-checks of the removed heads")
    parser.set_defaults(v21_checks=True)
    parser.add_argument("--encoding", choices=("v1", "v2", "v3", "v4", "v5", "v6"),
                        default="v6",
                        help="v1 = the run6 encoding; v2 adds action-"
                             "history tokens, exact prize/deck-position knowledge and a "
                             "learned card-id embedding (NEXT_MODEL_DESIGN.md); v3 adds "
                             "the de-aliased option surface + the remaining dropped state "
                             "fields (experiments/audit_2026_07_28/V3_BUILD_REPORT.md); "
                             "v4 = v3 inputs ++ MODEL-DECIDED multi-selection: the net picks "
                             "one option at a time and decides when to stop, so no code "
                             "answers a prompt that has more than one legal answer "
                             "(src/game/encode_selection.py). Same token width as v3; "
                             "v6 = v5 ++ two per-option chain-progress columns so a "
                             "BATCHED effect chain (Phantom Dive) stops being invisible "
                             "(src/game/state_encoder.py). Same token width as v5.")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--bf16", action="store_true",
                        help="bfloat16 autocast for the GPU learner")
    parser.add_argument("--no-overlap", dest="overlap", action="store_false",
                        help="disable the gen/train pipeline (next block generates on "
                             "the pool while the GPU trains; 1-iteration staleness)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the trunk's training entry points (needs triton; "
                             "triton-windows works -- validated 2026-07-23, ~1.2x integrated; "
                             "dynamic shapes for bucketed batches)")
    parser.add_argument("--gpu-server", action="store_true",
                        help="batch rollout forwards on a GPU inference-server process "
                             "(the d512+ unlock); workers fall back to local CPU on "
                             "any server hiccup")
    parser.add_argument("--no-worker-model", action="store_true",
                        help="workers build NO local model (needs --gpu-server): every "
                             "forward goes to the server, which frees the per-worker "
                             "model+trace so the pool can oversubscribe the cores; a "
                             "server failure retries twice then errors that decision")
    parser.add_argument("--games-per-worker", type=int, default=1,
                        help="games each worker keeps IN FLIGHT at once (request "
                             "pipelining): while one game waits on the GPU server the "
                             "worker steps/encodes the others. 1 = the old serial worker. "
                             "Per-game trajectories are unaffected (each game has its own "
                             "rng, engine battle and trackers); costs ~1 partly-finished "
                             "game's decision buffer of RAM per extra slot")
    parser.add_argument("--encode-cache", action="store_true",
                        help="v2/v3: use the vectorised encoder "
                             "(src/game/encode_history_cached.py). Byte-identical output, "
                             "~2.4x cheaper; certified at level 1 over 200 (v2) / "
                             "30 (v3) seeded games")
    parser.add_argument("--no-packed-transfer", dest="packed_transfer",
                        action="store_false", default=True,
                        help="send each decision's token/option matrices to the parent "
                             "DENSE instead of packed (default: packed). Packing is pure "
                             "transport -- assemble unpacks byte-identical arrays -- and "
                             "cuts one worker->parent pipe write ~5.4x, which is what "
                             "makes 36 workers x --games-per-worker 4 safe under the v3 "
                             "widths (see experiments/audit_2026_07_28/V3_SPEED_REPORT.md)")
    parser.add_argument("--eager-unpack", action="store_true",
                        help="unpack each packed matrix into its own dense array in "
                             "`assemble` (the pre-2026-07-29 placement) instead of "
                             "scattering it straight into the collated minibatch. Same "
                             "bytes; ~3x more parent-side work on the serial critical path")
    parser.add_argument("--pinned-staging", dest="pinned_staging",
                        action="store_true", default=True,
                        help="stage each minibatch through reusable page-locked buffers "
                             "so the prefetcher's H2D copy is a real overlapping DMA "
                             "(default ON; pure transport, bytes unchanged)")
    parser.add_argument("--no-pinned-staging", dest="pinned_staging",
                        action="store_false",
                        help="copy minibatches straight out of pageable memory (the "
                             "pre-07-28 behaviour)")
    parser.add_argument("--resign-threshold", type=float, default=0.95,
                        help="end a self-play game early once BOTH critics agree it is "
                             "decided: one seat <= -T and the other >= +T on "
                             "--resign-persist consecutive of their OWN decisions. "
                             "0 disables. Off for games with a snapshot or scripted seat")
    parser.add_argument("--resign-persist", type=int, default=6,
                        help="consecutive own-decisions each side must hold its verdict "
                             "for before the game is resigned")
    parser.add_argument("--resign-audit-fraction", type=float, default=0.10,
                        help="fraction of TRIGGERS that are played out to natural "
                             "completion instead of resigning, to measure how often the "
                             "'hopeless' seat still comes back (the safety gauge)")
    parser.add_argument("--past-fraction", type=float, default=0.2)
    parser.add_argument("--snapshot-every", type=int, default=25)
    parser.add_argument("--snapshots-keep", type=int, default=8)
    parser.add_argument("--probe-every", type=int, default=50)
    parser.add_argument("--probe-games", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--full-ckpt-every", type=int, default=0,
                        help="also keep a VERSIONED full checkpoint (weights/full_v<N>.pt) "
                             "every N iterations. --ckpt-every only rewrites one rolling "
                             "ppo_latest.pt, and snapshot_v<N>.pt is trunk weights ONLY, so "
                             "without this no earlier iteration is resumable: its aux heads, "
                             "payability head and Adam moments were never written. Costs "
                             "~+69 MB and ~0.04 s per save at d128 (measured 2026-08-08: "
                             "full 72.9 MB vs snapshot 4.0 MB; optimizer 40.5 + aux 27.1 of "
                             "it). 0 = off, the historical behaviour.")
    parser.add_argument("--deck-mode", choices=("corpus", "alakazam", "focus"),
                        default="corpus",
                        help="focus: seat 0 always plays --focus-deck, seat 1 samples "
                             "the corpus popularity-weighted (the real metagame mix)")
    parser.add_argument("--focus-deck",
                        default=str(ROOT / "submissions" / "submission_alakazam"
                                    / "deck.csv"))
    parser.add_argument("--opponent-bundle", default=None,
                        help="path to a submission bundle dir: seat 1 is played by its "
                             "agent (scorer-only via HYDRA_SEARCH=0) with its own deck; "
                             "snapshot opponents are disabled in this mode")
    parser.add_argument("--probe-bundle", default=None,
                        help="load a bundle for the 'specialist' PROBE only; generation "
                             "stays normal self-play (combine with --matchup-deck for "
                             "both-seats matchup training with a specialist benchmark)")
    parser.add_argument("--matchup-deck", default=None,
                        help="path to a deck.csv: seat 1 ALWAYS pilots it "
                             "(matchup-expert training)")
    parser.add_argument("--matchup-pool", default=None,
                        help="path to a data/decks/matchups/<archetype>/ directory: seat 1 "
                             "draws UNIFORMLY from that archetype's decklist pool, a fresh "
                             "list per game (matchup fine-tuning). Mutually exclusive with "
                             "--matchup-deck, which is the single-list version")
    parser.add_argument("--field-pools", default=None,
                        help="path to the data/decks/matchups/ ROOT: seat 1's deck is a "
                             "UNIFORM archetype followed by a UNIFORM list inside it, so "
                             "every archetype is an equally common opponent regardless of "
                             "ladder share or how many lists it has. Overrides --deck-mode's "
                             "popularity-weighted field; exclusive with --matchup-deck / "
                             "--matchup-pool")
    # --- search-in-the-loop generation (SEARCH_TRAINING_DESIGN.md items 1-15) --------- #
    parser.add_argument("--search-gen", action="store_true",
                        help="search-in-the-loop generation: a fraction of decisions are "
                             "FULL-SEARCHED (determinized PUCT over the engine, leaves "
                             "evaluated on the GPU server) and emit a pruned visit "
                             "distribution as the policy target. Off = every code path "
                             "below is dead and generation is byte-identical to today")
    parser.add_argument("--search-p", type=float, default=0.25,
                        help="probability that a single-pick decision is searched "
                             "(playout-cap randomization; a RANDOM gate, never a "
                             "confidence heuristic)")
    parser.add_argument("--sims", type=int, default=24,
                        help="simulations per searched decision (July band sweep knee)")
    parser.add_argument("--sims-deep", type=int, default=64,
                        help="the DEEP simulation budget, drawn with probability "
                             "--deep-p instead of --sims (mixed budgets)")
    parser.add_argument("--deep-p", type=float, default=0.05,
                        help="probability a searched decision draws --sims-deep")
    parser.add_argument("--trees-per-worker", type=int, default=4,
                        help="games/trees a worker interleaves under --search-gen (one "
                             "engine agent pointer each). This is the direct lever on the "
                             "realized leaf-batch size at the GPU server; overrides "
                             "--games-per-worker when --search-gen is on")
    parser.add_argument("--leaf-batch", type=int, default=8,
                        help="max leaf evaluations one tree keeps outstanding (virtual "
                             "loss covers the in-flight edges)")
    parser.add_argument("--k-forced", type=float, default=2.0,
                        help="forced-playout constant: every root child gets at least "
                             "sqrt(k * prior * visits) visits, which are then pruned back "
                             "out of the target unless search raised the child's value")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--prior-cap", type=float, default=0.95,
                        help="expand only the top children covering this prior mass")
    parser.add_argument("--dirichlet-scale", type=float, default=10.0,
                        help="root Dirichlet alpha = scale / num_children")
    parser.add_argument("--dirichlet-weight", type=float, default=0.25)
    parser.add_argument("--search-temp-turns", type=int, default=10,
                        help="play proportional to pruned visits for this many turns, "
                             "argmax after")
    parser.add_argument("--search-decided-threshold", type=float, default=0.9,
                        help="root |V| above this counts the position as decided")
    parser.add_argument("--search-decided-persist", type=int, default=2,
                        help="consecutive decided own-decisions after which search is "
                             "skipped (mirrors the resign gate; endgame sims teach little)")
    parser.add_argument("--no-lazy-expand", dest="lazy_expand", action="store_false",
                        help="featurize a leaf's options at its FIRST evaluation instead "
                             "of waiting until the node is actually entered again "
                             "(default: lazy; same tree, fewer option encodes)")
    parser.set_defaults(lazy_expand=True)
    parser.add_argument("--no-manual-coin", dest="manual_coin", action="store_false",
                        help="let the engine flip search coins instead of treating them "
                             "as 50/50 chance nodes")
    parser.set_defaults(manual_coin=True)
    parser.add_argument("--root-algo", choices=("puct", "gumbel"), default="puct",
                        help="ROOT algorithm. puct = prior cap + Dirichlet + forced "
                             "playouts + target pruning. gumbel = Gumbel-top-k sampling + "
                             "sequential halving + completed-Q target (Dirichlet and "
                             "forced playouts are OFF under gumbel by construction)")
    parser.add_argument("--gumbel-actions", type=int, default=8,
                        help="m: root actions Gumbel-sampled without replacement")
    parser.add_argument("--gumbel-sigma", type=float, default=1.0,
                        help="scale of sigma(q) = scale * (50 + max visits) * q")
    # --- expert-iteration training (--ei) -------------------------------------------- #
    parser.add_argument("--ei", action="store_true",
                        help="replace the PPO objective with expert iteration: CE toward "
                             "the pruned visit target on searched decisions + a KL anchor "
                             "to the frozen warm-start policy on all of them. The PPO path "
                             "is untouched and still the default")
    parser.add_argument("--anchor-weight", type=float, default=0.5,
                        help="--ei: weight of KL(current || frozen warm start)")
    parser.add_argument("--ppo-anchor-weight", type=float, default=0.0,
                        help="PPO path only: weight of KL(current || frozen loaded "
                             "policy) added to the PPO loss on every decision. The "
                             "consolidation-phase leash of interleaved PPO/EI training "
                             "(PPO warm-started from an EI checkpoint sharpens against "
                             "the real objective while the anchor defends the rare "
                             "search-taught behaviors). 0 (default) = no anchor is "
                             "built and plain PPO is byte-identical")
    parser.add_argument("--fresh-optimizer", action="store_true",
                        help="skip the optimizer state on --resume (phase transitions "
                             "of interleaved PPO/EI: a new objective's moments have "
                             "nothing to do with the previous phase's)")
    parser.add_argument("--search-value-weight", type=float, default=0.3,
                        help="--ei: blend the value target on SEARCHED decisions toward "
                             "the search root value (0 = pure outcome). Watched by the "
                             "per-block EV(out) diagnostic and killable mid-run through "
                             "<run>/ei_config.json")
    parser.add_argument("--rich-dropout", type=float, default=0.10,
                        help="--ei: fraction of TRAINING rows whose engine-effect (rich) "
                             "block is zeroed, so the value head stays calibrated with "
                             "and without it")
    parser.add_argument("--replay-window", type=int, default=2,
                        help="--ei: how many iterations of SEARCHED positions the replay "
                             "buffer keeps (window 2 x --epochs 2 = searched rows trained "
                             "~4 epochs, unsearched 2). 1 = no replay")
    parser.add_argument("--replay-priority", type=float, default=0.5,
                        help="--ei: blend of the replay sampling weight between uniform "
                             "(0) and normalized KL(visit target || policy at insertion) "
                             "(1)")
    parser.add_argument("--prize-labels", choices=("deduction", "truth"),
                        default="deduction",
                        help="v22 my_prizes / opp_prizes label source: the residual "
                             "DEDUCTION (default, what the audit certified) or "
                             "DumpTrueState per decision (fixes the Redeemable-Ticket "
                             "propagation defect; needs engine_src/build_truth/cg.dll). "
                             "LABELS ONLY -- never an input, a prior or a determinization")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--probe-only", action="store_true",
                        help="just run the panel probe with --probe-ckpt (baselines)")
    parser.add_argument("--probe-ckpt", default=None)
    args = parser.parse_args()
    if args.action_rules.strip():
        rules = [rule.strip() for rule in args.action_rules.split(",") if rule.strip()]
        # BEFORE any pool spawns: children inherit the env and enable on import.
        os.environ["CG_ACTION_RULES"] = ",".join(rules)
        state_encoder.enable_action_rules(*rules)
        print(f"ACTION RULES ENABLED (opt-in): {rules}", flush=True)
    # Caught here as well as in init_worker: a failing pool initializer only makes the pool
    # respawn workers and reprint the traceback forever, so the run has to die up front.
    assert not (args.no_worker_model and not args.gpu_server), \
        "--no-worker-model needs --gpu-server: no local model AND no server = no forward"
    # --heads v22 reads the engine state dump the ENCODER already decodes, by decoding it
    # once in the trainer and handing it down as rich_override. The cached encoder decodes
    # internally and rejects an override, so the two cannot be combined without a second
    # dump_state call per decision -- refuse instead of paying for it silently.
    assert not (args.heads in ("v22", "v23", "v24", "v25", "v26") and args.aux_weight > 0
                and args.encode_cache), \
        "--heads v22 is incompatible with --encode-cache (it would double-decode DumpState)"
    assert not (args.search_gen and args.encoding not in ("v5", "v6")), \
        "--search-gen is wired to the v5/v6 decision surface (single-pick selects)"
    assert not (args.ei and not args.search_gen), \
        "--ei has no policy teacher without --search-gen"
    # --search-gen makes tree concurrency the lever on leaf-batch size, so it names the
    # worker's in-flight game count itself instead of inheriting --games-per-worker.
    if args.search_gen:
        args.games_per_worker = max(1, args.trees_per_worker)
    search_settings = None
    if args.search_gen:
        search_settings = {
            "p": args.search_p, "sims": args.sims, "sims_deep": args.sims_deep,
            "deep_p": args.deep_p, "k_forced": args.k_forced, "c_puct": args.c_puct,
            "prior_cap": args.prior_cap, "dirichlet_scale": args.dirichlet_scale,
            "dirichlet_weight": args.dirichlet_weight,
            "max_outstanding": args.leaf_batch, "manual_coin": args.manual_coin,
            "lazy_expand": args.lazy_expand, "temp_turns": args.search_temp_turns,
            "decided_threshold": args.search_decided_threshold,
            "decided_persist": args.search_decided_persist,
            "root_algo": args.root_algo, "gumbel_actions": args.gumbel_actions,
            "gumbel_sigma": args.gumbel_sigma}
    if args.eager_unpack:
        global LAZY_UNPACK
        LAZY_UNPACK = False

    if args.heads in ("v23", "v24", "v25", "v26") and args.aux_weight > 0:
        _require_v23_engine()

    run_dir = ROOT / "runs" / args.run_name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    focus_deck = None
    if args.deck_mode == "focus":
        focus_deck = [int(line) for line in Path(args.focus_deck).read_text().split()
                      if line.strip()]
        assert len(focus_deck) == 60, f"focus deck has {len(focus_deck)} cards"
    matchup_deck = None
    if args.matchup_deck:
        matchup_deck = [int(line) for line in Path(args.matchup_deck).read_text().split()
                        if line.strip()]
        assert len(matchup_deck) == 60, f"matchup deck has {len(matchup_deck)} cards"
    matchup_pool = None
    if args.matchup_pool:
        assert not args.matchup_deck, "--matchup-pool and --matchup-deck are exclusive"
        pool_dir = Path(args.matchup_pool)
        matchup_pool = [[int(line) for line in path.read_text().split() if line.strip()]
                        for path in sorted(pool_dir.glob("*.csv"))]
        assert matchup_pool, f"no decklists in {pool_dir}"
        for path, deck in zip(sorted(pool_dir.glob("*.csv")), matchup_pool):
            assert len(deck) == 60, f"{path.name} has {len(deck)} cards"
        print(f"matchup pool: {len(matchup_pool)} lists from {pool_dir.name} "
              f"(uniform per game)", flush=True)
    field_pools = None
    if args.field_pools:
        assert not (args.matchup_deck or args.matchup_pool), \
            "--field-pools is exclusive with --matchup-deck / --matchup-pool"
        root = Path(args.field_pools)
        for directory in sorted(entry for entry in root.iterdir() if entry.is_dir()):
            decks = []
            for path in sorted(directory.glob("*.csv")):
                deck = [int(line) for line in path.read_text().split() if line.strip()]
                assert len(deck) == 60, f"{path} has {len(deck)} cards"
                decks.append(deck)
            if decks:
                field_pools = (field_pools or []) + [(directory.name, decks)]
        assert field_pools, f"no archetype pools under {root}"
        total = sum(len(decks) for _name, decks in field_pools)
        print(f"field pools: {len(field_pools)} archetypes / {total} lists from "
              f"{root} -- uniform archetype ({1 / len(field_pools):.3f} each), then "
              f"uniform list", flush=True)
        for name, decks in field_pools:
            print(f"  {name:18s} {len(decks):4d} lists  "
                  f"{1 / len(field_pools) / len(decks):.5f} per list", flush=True)

    if args.probe_only:
        assert args.probe_ckpt, "--probe-only needs --probe-ckpt"
        ckpt_encoding = torch.load(args.probe_ckpt, map_location="cpu",
                                   weights_only=False).get("encoding", "v1")
        with multiprocessing.Pool(args.workers, initializer=init_worker,
                                  initargs=(str(weights_dir), args.deck_mode,
                                            args.probe_ckpt, None, None, None,
                                            focus_deck, None, ckpt_encoding)) as pool:
            rates = run_probe(pool, None, args, seed_base=77000000)
        behavior = rates.pop("_behavior", None)
        print(f"probe {Path(args.probe_ckpt).name} ({args.deck_mode} decks): "
              + "  ".join(f"{name} {rate:.3f}" for name, rate in rates.items())
              + (f" | behaviour {behavior}" if behavior else ""), flush=True)
        return

    set_encoding(args.encoding)        # before ANY model is built: it sets the dims
    architecture = {"d_model": args.d_model, "num_layers": args.num_layers,
                    "num_heads": args.num_heads, "feedforward_dim": args.ff_dim}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.tf32 and device == "cuda":
        # TRAINING-STEP ONLY (spawned server/worker processes don't inherit this, so the
        # certified fp32 generation path is untouched). TF32 rounds matmul INPUTS to a
        # 10-bit mantissa with fp32 accumulation -- NOT a rerun of the banned bf16
        # autocast (8-bit mantissa through the whole forward, logit-scale corruption).
        # Canary: kl_first_eval measures the train-vs-generation forward gap live; if
        # TF32 drifts it, the number says so within one iteration.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = build_model(**architecture)
    v21_modules = tuple(name for name in
                        (part.strip() for part in args.v21_modules.split(","))
                        if name)
    assert set(v21_modules) <= {"token", "side", "action"}, \
        f"--v21-modules got {v21_modules}"
    aux = None
    aux_v23 = None
    if args.aux_weight > 0:
        if args.heads in ("v22", "v23", "v24", "v25", "v26"):
            aux = AuxHeadsV22(args.d_model, modules=v21_modules, suite=args.heads)
            if args.heads in ("v23", "v24", "v25", "v26"):
                # A SEPARATE module under its own checkpoint key, so a v22 checkpoint
                # still loads its aux_state intact and only these heads start fresh.
                aux_v23 = aux_head_labels.AuxHeadsV23(args.d_model, option_dim=OPTION_DIM,
                                                 suite=args.heads)
        elif args.heads == "v21":
            aux = AuxHeadsV21(args.d_model, modules=v21_modules)
        elif args.heads == "full":
            aux = AuxHeadsFull(args.d_model)
        else:
            aux = AuxHeads(args.d_model)
    payability = PayabilityHeads(args.d_model) if args.attack_aux_weight > 0 else None
    start_iteration = 0
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(warm_start_v5_into_v6(resume_state["state_dict"], model))
        if aux is not None and resume_state.get("aux_state"):
            try:
                aux.load_state_dict(resume_state["aux_state"])
            except RuntimeError:              # ckpt saved with a different --heads set
                print("aux state incompatible (heads changed?), starting aux fresh",
                      flush=True)
        if aux_v23 is not None and resume_state.get("aux_v23_state"):
            v23_state = resume_state["aux_v23_state"]
            # v23/v24 checkpoint resumed under --heads v25: the ONLY keys allowed to be
            # missing are the v25 fetch head's own -- everything the checkpoint has loads
            # intact and only fetch_delay starts fresh. Any other mismatch falls through
            # to the strict load and its loud fresh-start.
            v25_only = {"fetch_delay.weight", "fetch_delay.bias",
                        "fetch_scale", "fetch_seen",
                        "wasted.weight", "wasted.bias",
                        "wasted_scale", "wasted_seen"}
            missing = set(aux_v23.state_dict()) - set(v23_state)
            unexpected = set(v23_state) - set(aux_v23.state_dict())
            try:
                if missing and missing <= v25_only and not unexpected:
                    aux_v23.load_state_dict(v23_state, strict=False)
                    print("v23 aux state loaded; fetch_delay starts fresh (v25 upgrade)",
                          flush=True)
                else:
                    aux_v23.load_state_dict(v23_state)
            except RuntimeError:
                print("v23 aux state incompatible, starting those heads fresh",
                      flush=True)
        if payability is not None and resume_state.get("payability_state"):
            try:
                payability.load_state_dict(resume_state["payability_state"])
            except RuntimeError:              # ckpt saved at a different d_model
                print("payability state incompatible, starting those heads fresh",
                      flush=True)
        start_iteration = int(resume_state.get("iteration", 0))
        print(f"resumed {args.resume} at iteration {start_iteration}", flush=True)
    # Move to the device BEFORE creating/loading the optimizer: load_state_dict casts
    # Adam's state to the CURRENT param device, so loading while params sit on CPU and
    # moving afterwards leaves the state behind (cuda-vs-cpu crash on the first step).
    model.to(device)
    model.eval()
    if aux is not None:
        aux.to(device)
    if aux_v23 is not None:
        aux_v23.to(device)
    if payability is not None:
        payability.to(device)
    if args.compile:
        # torch 2.13 inductor bug: mix-order reduction codegen dies with CantSplit on
        # our dynamic-shape batches (s6*s97 + s97 not divisible by s97) -- turn it off.
        import torch._inductor.config as inductor_config
        if hasattr(inductor_config.triton, "mix_order_reduction"):
            inductor_config.triton.mix_order_reduction = False
        model.policy_value = torch.compile(model.policy_value, dynamic=True)
        model.policy_value_context = torch.compile(model.policy_value_context,
                                                   dynamic=True)
        model.policy_value_tokens = torch.compile(model.policy_value_tokens,
                                                  dynamic=True)
        print("torch.compile enabled on trunk entry points", flush=True)
    trained_parameters = list(model.parameters()) \
        + (list(aux.parameters()) if aux is not None else []) \
        + (list(payability.parameters()) if payability is not None else [])
    try:
        optimizer = torch.optim.Adam(trained_parameters, lr=args.lr,
                                     fused=(device == "cuda"))
    except (RuntimeError, TypeError):
        optimizer = torch.optim.Adam(trained_parameters, lr=args.lr)
    # The v23 param group must exist BEFORE the load when the checkpoint already has it,
    # and only AFTER when it does not (2026-08-06 audit). load_state_dict matches param
    # groups positionally and raises if the COUNT differs -- and unlike Module.load_state_dict
    # it applies nothing before raising, so the old always-after ordering meant every resume
    # of a v23 run hit `ValueError: loaded state dict has a different number of parameter
    # groups`, printed a reassuring message and silently restarted ALL Adam moments from
    # zero at lr 2.5e-4. It fired 6 times in d256_v6 (iterations 321/341/491/511/551/641).
    # Adding it first when the shapes already match keeps the moments; adding it last when
    # they do not preserves the documented --heads v22 -> v23 upgrade path.
    resume_groups = len(((resume_state or {}).get("optimizer") or {})
                        .get("param_groups", ()))
    # v25 splits the v23 module across TWO groups -- group 1 the base heads (in the exact
    # parameter order v23/v24 always used, so a v24 checkpoint's moments map positionally),
    # group 2 the fetch_delay head alone, so resuming a pre-v25 checkpoint keeps every
    # existing moment and only the new head starts fresh. Group order is deterministic on
    # every path: [main, base_v23, fetch].
    base_v23_parameters = list(aux_v23.base_parameters()) if aux_v23 is not None else []
    v25_parameters = list(aux_v23.v25_parameters()) if aux_v23 is not None else []
    if aux_v23 is not None and resume_groups >= 2:
        optimizer.add_param_group({"params": base_v23_parameters, "lr": args.lr})
        trained_parameters = trained_parameters + base_v23_parameters
    if v25_parameters and resume_groups == 3:
        optimizer.add_param_group({"params": v25_parameters, "lr": args.lr})
        trained_parameters = trained_parameters + v25_parameters
    if resume_state is not None and "optimizer" in resume_state and not args.ei \
            and not args.fresh_optimizer:
        try:
            optimizer.load_state_dict(resume_state["optimizer"])
        except ValueError as error:           # bare ckpt resumed with aux (new params)
            # Print the reason: a discarded optimizer must never look like a benign note.
            print(f"OPTIMIZER STATE DISCARDED -- all Adam moments restart from zero "
                  f"({error})", flush=True)
        # load_state_dict restores each param group's lr FROM THE CHECKPOINT, silently
        # overriding --lr. A checkpoint saved at the end of an LR-annealed run carries
        # lr=0.0, so a fine-tune resumed from it trains every parameter at lr 0 -- the
        # 2026-08-11 d128b fine-tune batch ran all 8 archetypes as exact no-ops this
        # way. The anneal is stateless (recomputed from the iteration each update), so
        # the command line is always the authority: reset lr after loading the moments.
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    if aux_v23 is not None and len(optimizer.param_groups) == 1:
        # Not yet added above: a v22 checkpoint being resumed with v23 heads, or a fresh
        # run. Its own group keeps every pre-existing moment where it was and gives only
        # the new heads a fresh state.
        optimizer.add_param_group({"params": base_v23_parameters, "lr": args.lr})
        trained_parameters = trained_parameters + base_v23_parameters
    if v25_parameters and len(optimizer.param_groups) == 2:
        # The v25 heads' group was not loadable from the checkpoint (pre-v25 or fresh):
        # add it now with fresh moments.
        optimizer.add_param_group({"params": v25_parameters, "lr": args.lr})
        trained_parameters = trained_parameters + v25_parameters
    # --ei warm start: trunk + policy + value + the aux suites load above; the OPTIMIZER
    # starts fresh (a new objective's moments have nothing to do with PPO's), and the
    # policy the KL anchor leashes to is a FROZEN COPY of exactly what was loaded.
    anchor = None
    if (args.ei and args.anchor_weight > 0) \
            or (not args.ei and args.ppo_anchor_weight > 0):
        anchor = build_model(**architecture)
        anchor.load_state_dict({key: value.cpu()
                                for key, value in model.state_dict().items()})
        anchor.to(device).eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
        print(f"{'EI' if args.ei else 'PPO consolidation'}: "
              "frozen anchor policy taken at load", flush=True)

    parameters = sum(p.numel() for p in model.parameters())
    print(f"PPO self-play | encoding {ENCODING} | trunk {parameters:,} params "
          f"(tok {TOKEN_DIM} glob {GLOBAL_DIM} opt {OPTION_DIM} zones {ZONE_COUNT}"
          f"{f' card-emb {MODEL_CARD_VOCAB}x{model.config.card_embedding_dim}' if MODEL_CARD_VOCAB else ''}) "
          f"| {args.workers} workers | {args.games_per_iter} games/iter | "
          f"decks={args.deck_mode} | device {device}", flush=True)

    def publish_weights(version):
        cpu_state = {key: value.cpu() for key, value in model.state_dict().items()}
        torch.save(cpu_state, weights_dir / f"current_v{version}.pt")
        stale = weights_dir / f"current_v{version - 2}.pt"
        if stale.exists():
            stale.unlink()
        return cpu_state

    snapshots = sorted(int(p.stem.split("_v")[1])
                       for p in weights_dir.glob("snapshot_v*.pt"))
    retired = []                       # pruned versions whose file survives one more iteration
    metrics_path = run_dir / "metrics.jsonl"
    rng = random.Random(args.seed_base + 12345)
    from collections import deque
    replay = deque(maxlen=max(1, args.replay_window - 1))   # PAST iterations only
    cumulative_games = 0
    start_time = time.time()

    publish_weights(start_iteration)
    published_version = start_iteration

    server_process = None
    server_queues = None
    worker_counter = None
    if args.gpu_server:
        assert device == "cuda", "--gpu-server needs a GPU"
        context = multiprocessing.get_context("spawn")
        request_queue = context.Queue()
        # Spare queues: the worker index counts UP, so a pool worker that dies and is
        # respawned takes the next free slot. Without the spares it would index past the
        # end and the pool would respawn-and-crash forever (reusing a live worker's slot
        # by modulo would be worse: two workers reading one reply stream).
        reply_queues = [context.Queue() for _ in range(args.workers + 16)]
        server_queues = (request_queue, reply_queues)
        worker_counter = context.Value("i", 0)
        server_process = context.Process(
            target=inference_server,
            args=(request_queue, reply_queues, str(weights_dir), architecture,
                  args.encoding, 64, args.tf32),
            daemon=True)
        server_process.start()
        print(f"GPU inference server up (pid {server_process.pid})", flush=True)

    def build_tasks(iteration):
        tasks = []
        for i in range(args.games_per_iter):
            seed = args.seed_base + 1000000 + (iteration - 1) * args.games_per_iter + i
            snapshot_version, snapshot_side = None, 0
            if (snapshots and not args.opponent_bundle
                    and rng.random() < args.past_fraction):
                snapshot_version = rng.choice(snapshots)
                snapshot_side = rng.randrange(2)
            tasks.append(("selfplay", seed, published_version, snapshot_version,
                          snapshot_side))
        return tasks

    with multiprocessing.Pool(args.workers, initializer=init_worker,
                              initargs=(str(weights_dir), args.deck_mode, None,
                                        architecture, server_queues, worker_counter,
                                        focus_deck, matchup_deck, args.encoding,
                                        args.no_worker_model,
                                        args.opponent_bundle, args.probe_bundle,
                                        args.encode_cache,
                                        args.games_per_worker,
                                        args.resign_threshold, args.resign_persist,
                                        args.resign_audit_fraction,
                                        args.packed_transfer,
                                        args.attack_aux_weight > 0,
                                        args.heads, aux is not None, v21_modules,
                                        args.v21_checks, search_settings,
                                        args.prize_labels, matchup_pool,
                                        field_pools)) as pool:
        def dispatch(iteration):
            """The generation block for `iteration` as an AsyncResult. With
            --games-per-worker K the unit of work is a CHUNK of K games so one worker can
            keep K of them in flight; K=1 is the old one-task-per-call mapping."""
            tasks = build_tasks(iteration)
            if args.games_per_worker <= 1:
                return pool.map_async(run_task, tasks, chunksize=1)
            block = args.games_per_worker
            chunks = [tasks[start:start + block]
                      for start in range(0, len(tasks), block)]
            return pool.map_async(run_task_block, chunks, chunksize=1)

        def collect(async_result):
            records = async_result.get()
            if args.games_per_worker <= 1:
                return records
            return [record for chunk in records for record in chunk]

        pending = None                    # (iteration, AsyncResult) pipelined gen block
        # Captured ONCE so the ent anneal always interpolates from the launch value:
        # the block below writes the annealed coefficient back into args.ent_coef
        # (ppo_update reads it there), which would otherwise corrupt the next
        # iteration's interpolation base.
        ent_coef_base = args.ent_coef
        for iteration in range(start_iteration + 1, args.iterations + 1):
            # True wall clock for the whole iteration. The phase timers below do NOT sum
            # to it (gen is only the WAIT, and probe/snapshot/checkpoint are untimed), so
            # the log prints this explicitly -- a manual stopwatch once disagreed with the
            # phase numbers by 3x and the parenthetical (asm col gpu) reads like a
            # breakdown of `train` when only col/gpu are inside it.
            iteration_start = time.time()
            if pending is not None and pending[0] == iteration:
                async_result = pending[1]
            else:
                async_result = dispatch(iteration)
            wait_start = time.time()
            records = collect(async_result)
            generation_seconds = time.time() - wait_start   # time WAITED (overlap hides the rest)
            # Pipeline: generate the NEXT block on the pool while the GPU trains this
            # one. Those games play with the current (soon 1-iteration-stale) weights;
            # the PPO ratio is anchored to the stored behavior logprobs, so this is
            # algorithmically clean. Skipped when this iteration ends with a probe --
            # the probe needs the pool free.
            pending = None
            if (args.overlap and iteration < args.iterations
                    and not (args.probe_every and iteration % args.probe_every == 0)):
                pending = (iteration + 1, dispatch(iteration + 1))

            assemble_start = time.time()
            label_stats = Counter()
            flat = assemble(records, args.gamma, args.lam, heads=args.heads,
                            aux_labels=aux is not None, v21_modules=v21_modules,
                            v21_checks=args.v21_checks, stats=label_stats,
                            payability=payability is not None, search=args.search_gen)
            assemble_seconds = time.time() - assemble_start
            if not flat:
                print(f"iter {iteration}: no decisions collected, skipping", flush=True)
                continue
            replayed = 0
            if args.ei:
                # --- recency-windowed replay of SEARCHED positions. A searched row is far
                # more expensive than an unsearched one and its target does not go stale
                # the way a PPO ratio does (the visit distribution is a fixed teacher
                # label), so it is trained over --replay-window iterations while the
                # unsearched rows are seen once. NO importance weights: this is
                # distillation toward a fixed target, not value bootstrapping, so a
                # non-uniform sampler biases WHICH targets are rehearsed and nothing else.
                searched_rows = [row for row in flat if row.get("searched")]
                if replay and args.replay_window > 1:
                    pool_rows = [row for block in replay for row in block]
                    if pool_rows:
                        weights = np.array([row.get("search_kl") or 0.0
                                            for row in pool_rows], dtype=np.float64)
                        total = weights.sum()
                        weights = (weights / total if total > 0
                                   else np.full(len(pool_rows), 1.0 / len(pool_rows)))
                        uniform = np.full(len(pool_rows), 1.0 / len(pool_rows))
                        blend = ((1.0 - args.replay_priority) * uniform
                                 + args.replay_priority * weights)
                        blend = blend / blend.sum()
                        take = min(len(pool_rows), max(1, len(searched_rows)))
                        picked = np.random.default_rng(
                            args.seed_base + iteration).choice(
                            len(pool_rows), size=take, replace=False, p=blend)
                        flat = flat + [pool_rows[int(index)] for index in picked]
                        replayed = int(take)
                if args.replay_window > 1:
                    replay.append(searched_rows)
            values = np.array([d["value"] for d in flat], dtype=np.float32)
            returns = np.array([d["return"] for d in flat], dtype=np.float32)
            outcomes = np.array([d["outcome"] for d in flat], dtype=np.float32)
            ev_return = explained_variance(values, returns)
            ev_outcome = explained_variance(values, outcomes)

            train_start = time.time()
            if args.ei:
                # Live kill switch for the watched value blend (addendum 11): the run reads
                # <run>/ei_config.json every iteration, so the flag can be turned off
                # mid-run without a restart when the hedging signature shows up.
                override = run_dir / "ei_config.json"
                if override.exists():
                    try:
                        for key, value in json.loads(
                                override.read_text(encoding="utf-8")).items():
                            if hasattr(args, key):
                                setattr(args, key, value)
                    except Exception:
                        pass
                update = ei_update(model, optimizer, flat, args, device, aux=aux,
                                   payability=payability, anchor=anchor,
                                   aux_v23=aux_v23)
            else:
                if args.lr_anneal_end > args.lr_anneal_start:
                    span = args.lr_anneal_end - args.lr_anneal_start
                    progress = (iteration - args.lr_anneal_start) / span
                    progress = min(1.0, max(0.0, progress))
                    annealed_lr = args.lr + (args.lr_final - args.lr) * progress
                    for group in optimizer.param_groups:
                        group["lr"] = annealed_lr
                if args.ent_anneal_end > args.ent_anneal_start:
                    span = args.ent_anneal_end - args.ent_anneal_start
                    progress = (iteration - args.ent_anneal_start) / span
                    progress = min(1.0, max(0.0, progress))
                    args.ent_coef = (ent_coef_base
                                     + (args.ent_final - ent_coef_base) * progress)
                update = ppo_update(model, optimizer, flat, args, device, aux=aux,
                                    payability=payability, anchor=anchor,
                                    aux_v23=aux_v23)
            train_seconds = time.time() - train_start
            publish_weights(iteration)

            pure = [r for r in records if r["snapshot_side"] is None]
            versus_past = [r for r in records if r["snapshot_side"] is not None]
            p0_wins = float(np.mean([r["outcome"] > 0 for r in pure])) if pure else 0.0
            draws = float(np.mean([r["outcome"] == 0 for r in records]))
            past_wins = float(np.mean(
                [(r["outcome"] > 0) == (r["snapshot_side"] == 1)
                 for r in versus_past])) if versus_past else float("nan")
            # SEAT-SPLIT vs-past (fine-tune gauge, 2026-08-11): the snapshot-on-seat-1
            # subset is CURRENT model piloting OUR deck vs a FROZEN snapshot piloting the
            # field deck -- in a single-pool fine-tune that is exactly "the specialist vs
            # the base on the target matchup", free of the both-seats-learning confound
            # that makes p0 uninterpretable (measured 08-10: three fine-tunes' p0 FELL
            # while absolute strength rose because the net learned the opponent's deck
            # faster than it learned to beat it).
            opp_side = [r for r in versus_past if r["snapshot_side"] == 1]
            past_opp_wins = float(np.mean([r["outcome"] > 0 for r in opp_side])) \
                if opp_side else float("nan")
            mean_moves = float(np.mean([r["moves"] for r in records]))
            errors = sum(r["errors"] for r in records)
            # v4 no-auto-answer accounting: `forwards` = prompts the MODEL answered (one per
            # sub-pick), `forced*` = prompts with exactly one legal answer. Anything else
            # answering a prompt would show up as a gap between selects and forced+forwards.
            v4_totals = Counter()
            search_totals = Counter()
            for record in records:
                v4_totals.update(record.get("v4") or {})
                search_totals.update(record.get("search") or {})
            v4_line = {key: value / max(1, len(records))
                       for key, value in sorted(v4_totals.items())}
            # Resignation accounting. comeback_count / audit_count is the SAFETY GAUGE:
            # the audited slice played on after the trigger, so a comeback there is a game
            # the threshold would have mis-called. Iono / Counter Catcher make late swings
            # real, so watch it -- >5-8% means the threshold is too loose.
            resign_meta = [r.get("resign") or {} for r in records]
            resign_fraction = float(np.mean([bool(m.get("resigned"))
                                             for m in resign_meta])) if records else 0.0
            audit_count = sum(1 for m in resign_meta if m.get("audit"))
            comeback_count = sum(1 for m in resign_meta if m.get("comeback"))
            cumulative_games += len(records)
            games_per_second = cumulative_games / (time.time() - start_time)

            if args.snapshot_every and iteration % args.snapshot_every == 0:
                torch.save({key: value.cpu() for key, value
                            in model.state_dict().items()},
                           weights_dir / f"snapshot_v{iteration}.pt")
                snapshots.append(iteration)
                # DEFERRED unlink (TRAINING audit MAJOR-2): `dispatch(iteration + 1)` above
                # has already sampled snapshot versions from the pre-prune list, so deleting
                # a file here can make an IN-FLIGHT game fail to load its frozen seat --
                # which silently converts that seat to the current model and appends its
                # decisions to training data while the record still says "vs past". Retiring
                # a version one iteration late costs one extra ~2.4 MB file and closes it.
                while len(snapshots) > args.snapshots_keep:
                    retired.append(snapshots.pop(0))
                while len(retired) > 1:              # only versions no dispatch can name
                    (weights_dir / f"snapshot_v{retired.pop(0)}.pt").unlink(missing_ok=True)

            probe = None
            behavior = None
            if args.probe_every and iteration % args.probe_every == 0:
                # 2026-08-11 audit: an uncaught probe exception killed the iteration
                # BEFORE both checkpoint writes below -- an eval-only failure must
                # never cost training state.
                try:
                    probe = run_probe(pool, iteration, args,
                                      seed_base=88000000 + iteration * 10000)
                    behavior = probe.pop("_behavior", None)
                except Exception as error:
                    print(f"PROBE FAILED at iteration {iteration} "
                          f"(training continues): {error!r}", flush=True)
                    probe = None

            line = {"iteration": iteration, "games": len(records),
                    "decisions": len(flat), "generation_seconds": generation_seconds,
                    "train_seconds": train_seconds,
                    "assemble_seconds": assemble_seconds, "ev_return": ev_return,
                    "ev_outcome": ev_outcome, "p0_win": p0_wins, "draws": draws,
                    "vs_past_win": past_wins, "vs_past_opp_win": past_opp_wins,
                    "vs_past_opp_n": len(opp_side), "mean_moves": mean_moves,
                    "errors": errors, "games_per_second": games_per_second,
                    "resign_fraction": resign_fraction, "audit_count": audit_count,
                    "comeback_count": comeback_count, "v4_per_game": v4_line,
                    "search": dict(search_totals), "replayed": replayed,
                    "behavior": behavior, "lr": optimizer.param_groups[0]["lr"],
                    "ent_coef": args.ent_coef,
                    **update, "probe": probe}
            with open(metrics_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")
            probe_text = "" if probe is None else " | probe " + " ".join(
                f"{name} {rate:.2f}" for name, rate in probe.items())
            if behavior:
                # Behavioural counters (the actual point of the fine-tune): per probe game.
                probe_text += (f" | beh abl {behavior.get('abilities', 0):.2f} "
                               f"att {behavior.get('attachments', 0):.2f} "
                               f"surplus {behavior.get('surplus_attachments', 0):.2f}")
                if behavior.get("dark_top"):
                    probe_text += " dark " + ",".join(
                        f"{int(card)}:{rate:.2f}" for card, rate in behavior["dark_top"])
            if args.heads in ("v22", "v23", "v24", "v25", "v26"):
                # The v2.2 suite reports a metric AND a base rate for every component, so
                # the flat "every aux_* key" block would be ~66 numbers. Losses stay in
                # aux[...]; the quality pairs get their own q[...] block, `metric/base`.
                aux_text = "".join(
                    f" {name} {update[f'aux_{name}']:.2f}" for name in V22_COMPONENTS
                    if f"aux_{name}" in update)
                quality = []
                for name in V22_COMPONENTS:
                    metric = next((update[f"aux_{name}_{suffix}"]
                                   for suffix in ("acc", "recall", "mae")
                                   if f"aux_{name}_{suffix}" in update), None)
                    if metric is None:
                        continue
                    # 3 significant figures, not 2 decimals: the count-class base rates are
                    # ~1e-3 (one card played out of a 1268 vocabulary) and would print 0.00
                    quality.append(f"{name} {metric:.3g}/"
                                   f"{update.get(f'aux_{name}_base', float('nan')):.3g}")
                for name in aux_head_labels.V23_COMPONENTS + ("fetch_delay", "wasted_counters"):
                    # ...and the v23 suite reports in exactly the same shape
                    # (fetch_delay is v25-only and outside V23_COMPONENTS -- the buffer-
                    # sizing rule -- so it is appended here; its guards no-op elsewhere)
                    if f"aux_{name}" in update:
                        aux_text += f" {name} {update[f'aux_{name}']:.2f}"
                    metric = next((update[f"aux_{name}_{suffix}"]
                                   for suffix in ("acc", "recall")
                                   if f"aux_{name}_{suffix}" in update), None)
                    if metric is not None:
                        quality.append(f"{name} {metric:.3g}/"
                                       f"{update.get(f'aux_{name}_base', float('nan')):.3g}")
                # bench_bits is a POOLED recall over five different bits; print them apart
                # so a bit that never learns cannot hide behind the average. Metrics only:
                # these are not V23_COMPONENTS and carry no loss term of their own.
                for name in aux_head_labels.V23_L3_BIT_NAMES:
                    key = f"aux_benchbit_{name}_recall"
                    if key in update:
                        quality.append(f"bench:{name} {update[key]:.3g}/"
                                       f"{update.get(f'aux_benchbit_{name}_base', float('nan')):.3g}")
                if quality:
                    aux_text += "] | q[" + " ".join(quality)
            else:
                aux_text = "".join(f" {key[4:]} {value:.2f}" for key, value
                                   in sorted(update.items()) if key.startswith("aux_"))
            if label_stats.get("v21_check_supporter_n"):
                # Layer-3 supporter cross-check: a RATE, not an assert (a supporter can
                # reach the discard by an effect instead of by being played).
                aux_text += (f" supporter_miss "
                             f"{label_stats['v21_check_supporter_bad'] / label_stats['v21_check_supporter_n']:.2f}")
            for side in ("my", "opp"):
                # v2.2: how often the prize RESIDUAL derivation failed to close (masked).
                total = label_stats.get(f"v22_check_{side}_prize_n")
                if total:
                    aux_text += (f" {side}_prize_masked "
                                 f"{label_stats.get(f'v22_check_{side}_prize_masked', 0) / total:.2f}")
            if aux_text:
                probe_text = f" | aux[{aux_text.strip()}]" + probe_text
            if args.heads in ("v23", "v24", "v25", "v26"):
                # v2.3 LABEL TELEMETRY. Every one of these counters was being produced at
                # full scale and read by NOTHING: `label_stats` was consulted only for the
                # v21 supporter check and the two v22 prize checks, so "56% of chosen ATTACH
                # rows carry no label" and "82% of already_covered is masked" were invisible
                # for the whole of d256_v6 -- and so was every v23_bad_* tripwire
                # (2026-08-06 audit). Coverage first, then the integrity counters, which
                # must all stay at zero.
                coverage = " ".join(
                    f"{name[4:]} {label_stats[name]}" for name in
                    ("v23_attach_rows", "v23_attach_masked", "v23_attach_waste",
                     "v23_covered_masked", "v23_ko_delay_rows", "v23_decisive_rows",
                     "v23_bench_rows", "v23_prize_rows", "v23_fetch_rows",
                     "v23_wasted_rows",
                     "v23_attack_no_engine_state", "v23_ability_no_engine_state",
                     "v23_attach_deferred", "v23_attach_deferred_unresolved",
                     "v23_cost_tripwire", "v23_joint_fill_infeasible",
                     "v23_join_vs_engine")
                    if label_stats.get(name))
                bad = " ".join(f"{name[4:]} {count}" for name, count
                               in sorted(label_stats.items())
                               if name.startswith("v23_bad_") and count)
                terms = " ".join(
                    f"{key[4:]} {update[key]:.1f}" for key in
                    ("aux_v22_terms", "aux_v23_terms") if key in update)
                block = " ".join(part for part in (coverage, terms, bad) if part)
                if block:
                    probe_text += f" | v23[{block}]"
            if payability is not None:
                # acc/pos per head, attack columns (atk) then ability columns (abl): masked
                # accuracy against the label's own positive rate -- what a constant
                # predictor would score, i.e. the bar these heads have to clear.
                probe_text = "".join(
                    f" | {prefix} " + " ".join(
                        f"{name} {update.get(f'{prefix}_{name}_acc', float('nan')):.2f}/"
                        f"{update.get(f'{prefix}_{name}_pos', float('nan')):.2f}"
                        for name in ("now", "next", "hor"))
                    for prefix in ("atk", "abl")) + probe_text
            resign_text = "" if args.resign_threshold <= 0 else \
                f" | rsn {resign_fraction:.2f} cb {comeback_count}/{audit_count}"
            # v4: per game, how many prompts the MODEL answered vs how many had a single
            # legal answer. `fwd` counts sub-picks (a discard-3 select is 3+ of them).
            v4_text = "" if not v4_line else (
                f" | v4 sel {v4_line.get('selects', 0):.1f} fwd "
                f"{v4_line.get('forwards', 0):.1f} pick {v4_line.get('picks', 0):.1f} "
                f"stop {v4_line.get('stop', 0):.1f} forced "
                f"{v4_line.get('forced', 0) + v4_line.get('forced_pick', 0):.1f}")
            if v4_line.get("counter_masked"):
                # Opt-in counter_cap rule: options hidden per game (metrics already carry
                # it inside v4_per_game -- the row is written before this line runs).
                v4_text += f" cmask {v4_line['counter_masked']:.1f}"
            resign_text += v4_text
            if args.search_gen:
                # The search dial's own telemetry: how many decisions were searched, what
                # they cost, and -- the --trees-per-worker lever's effect -- the realized
                # mean LEAF BATCH the workers handed the GPU server.
                searched = search_totals.get("searched", 0)
                batches = max(1, search_totals.get("batches", 0))
                resign_text += (
                    f" | search {searched}/{search_totals.get('searchable', 0)} "
                    f"sims {search_totals.get('sims', 0) / max(1, searched):.0f} "
                    f"leafbatch {search_totals.get('leaves', 0) / batches:.1f} "
                    f"enc {search_totals.get('encodes', 0) / max(1, searched):.1f}/"
                    f"{search_totals.get('option_encodes', 0) / max(1, searched):.1f} "
                    f"hit {search_totals.get('cache_hits', 0) / max(1, searched):.1f} "
                    f"deep {search_totals.get('deep', 0)} "
                    f"rej {search_totals.get('rejected', 0)}")
            if args.ei:
                head = (f"pol {update['policy_loss']:.3f} "
                        f"anch {update['anchor_kl']:.4f} "
                        f"srch {update['searched_fraction']:.2f} "
                        f"rep {replayed} | "
                        f"EV(out) s {update['ev_out_searched']:.3f} "
                        f"u {update['ev_out_unsearched']:.3f}")
            else:
                head = (f"EV(ret) {ev_return:.3f} EV(out) {ev_outcome:.3f} | "
                        f"ent {update['entropy']:.2f} kl {update['kl']:.4f} "
                        f"clip {update['clipfrac']:.2f}")
                if "anchor_kl" in update:
                    head += f" anch {update['anchor_kl']:.4f}"
            print(f"iter {iteration} | {len(records)}g {len(flat)}d | "
                  f"WALL {time.time() - iteration_start:.0f}s "
                  f"[genwait {generation_seconds:.0f} asm {assemble_seconds:.0f} "
                  f"train {train_seconds:.0f} (col {update['collate_seconds']:.0f} "
                  f"gpu {update['gpu_seconds']:.0f})] | "
                  f"{head} | vloss {update['value_loss']:.3f} | "
                  f"p0 {p0_wins:.2f} draws {draws:.2f} len {mean_moves:.0f} | "
                  f"vs-past {past_wins:.2f} | {games_per_second:.1f} g/s"
                  f"{resign_text}{probe_text}", flush=True)

            published_version = iteration
            if iteration % args.ckpt_every == 0:
                save_checkpoint(run_dir / "ppo_latest.pt", model, optimizer,
                                iteration, args, aux=aux, payability=payability,
                                aux_v23=aux_v23)
            # --full-ckpt-every: the RESUMABLE archive. Kept versioned and never pruned --
            # the whole point is that a checkpoint chosen later (by eval, or after a run
            # regresses) can be resumed rather than wrapped, which would restart every aux
            # head and Adam moment on top of a trained trunk. Distinct filename prefix so
            # it cannot collide with snapshot_v*/current_v*, which the frozen-opponent and
            # weight-refresh paths glob for by name.
            if args.full_ckpt_every and iteration % args.full_ckpt_every == 0:
                save_checkpoint(weights_dir / f"full_v{iteration}.pt", model, optimizer,
                                iteration, args, aux=aux, payability=payability,
                                aux_v23=aux_v23)
    save_checkpoint(run_dir / "ppo_latest.pt", model, optimizer, args.iterations, args,
                    aux=aux, payability=payability, aux_v23=aux_v23)
    if server_process is not None:
        server_process.terminate()


if __name__ == "__main__":
    main()
