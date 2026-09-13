"""Token segment ids for the game-state transformer encoding.

Shared by the encoder (src/game/encode.py) and the model (src/models/transformer.py)
so the two can't drift. Each token carries an owner id and a zone id; these index the
model's owner/zone embedding tables, so NUM_OWNERS / NUM_ZONES size those tables.
"""

NUM_OWNERS = 3
OWNER_ME = 0
OWNER_OPPONENT = 1
OWNER_NEUTRAL = 2          # global/CLS token (and stadium) -- belong to no one

NUM_ZONES = 9
ZONE_GLOBAL = 0           # the CLS / global token
ZONE_ACTIVE = 1
ZONE_BENCH = 2
ZONE_HAND = 3
ZONE_DECK = 4
ZONE_DISCARD = 5
ZONE_PREDICTED = 6       # a card we BELIEVE is in the opponent's hidden cards (deck/hand/prizes)
ZONE_ATTACK = 7          # a capability token: one specific attack of an in-play Pokemon
ZONE_ABILITY = 8         # a capability token: one specific ability of an in-play Pokemon
