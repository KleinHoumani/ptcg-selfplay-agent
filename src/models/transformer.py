"""A small set-transformer over the board state (value net; policy stubbed).

One token per card (active / bench / hand) + a learned global/CLS token that carries the
scalar game features; the global token's output drives the value head. The board is a
*set*, so structure is injected with learned owner + zone embeddings rather than positional
order (a bench of {A, B} must equal {B, A}). Kept tiny for CPU eval.

forward inputs (produced by src/game/encode.py's vectorize step):
  token_features  [B, T, F]   per-card feature vectors (incl. the text embedding)
  owner_ids       [B, T]      OWNER_* per card token
  zone_ids        [B, T]      ZONE_* per card token
  padding_mask    [B, T]      True where a slot is padding (ignored by attention)
  global_features [B, G]      scalar game state for the global token
"""

from dataclasses import dataclass

import torch
from torch import nn

from src.segments import (
    NUM_OWNERS, NUM_ZONES,
    OWNER_ME, OWNER_OPPONENT, OWNER_NEUTRAL,
    ZONE_GLOBAL, ZONE_ACTIVE,
)


@dataclass
class GameStateTransformerConfig:
    token_feature_dim: int          # F: width of each card token's feature vector
    global_feature_dim: int         # G: width of the global scalar vector
    option_feature_dim: int = 0     # width of an option's feature vector (0 -> value head only)
    d_model: int = 128
    num_layers: int = 3
    num_heads: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.0
    num_zones: int = NUM_ZONES      # zone-embedding size; NUM_ZONES_FULL for encode_full models
                                    # (default keeps every existing checkpoint's shape)
    card_vocab: int = 0             # >0 adds a LEARNED per-card identity embedding, concatenated
                                    # to each token's features before the input projection
                                    # (v2 encoders emit card_ids). 0 = off, which is what every
                                    # existing checkpoint was built with -- their card_projection
                                    # keeps its original input width and loads unchanged.
    card_embedding_dim: int = 64     # ignored when card_vocab == 0


class GameStateTransformer(nn.Module):
    def __init__(self, config: GameStateTransformerConfig):
        super().__init__()
        self.config = config

        # Learned card identity, AUGMENTING the frozen text embedding already inside each
        # token's features (shared weights cannot add information the input lacks: two cards
        # that collide in the PCA-compressed text space are otherwise indistinguishable).
        # Built only when asked, so checkpoints trained without it keep their exact shapes.
        self.card_embedding = nn.Embedding(config.card_vocab, config.card_embedding_dim) \
            if config.card_vocab else None
        projection_width = config.token_feature_dim + (config.card_embedding_dim
                                                       if config.card_vocab else 0)

        # Separate input projections: all card-like tokens share one; the global token
        # (different schema) gets its own. Both land in the shared d_model.
        self.card_projection = nn.Linear(projection_width, config.d_model)
        self.global_projection = nn.Linear(config.global_feature_dim, config.d_model)

        # Learned structure, added (not concatenated). No positional order within a zone.
        self.owner_embedding = nn.Embedding(NUM_OWNERS, config.d_model)
        self.zone_embedding = nn.Embedding(config.num_zones, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,            # pre-norm: steadier when training from scratch
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers,
                                             enable_nested_tensor=False)

        # Value head off the global/CLS token (it attends over the whole board).
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )

        # Policy head: score each option from the attended board context (CLS) + the option's
        # own features. Built only when option_feature_dim is set (the actor-critic policy);
        # left None for value-net-only use.
        self.policy_score = nn.Sequential(
            nn.Linear(config.d_model + config.option_feature_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        ) if config.option_feature_dim else None

        # Q head (2026-08-04): a SECOND readout of the option scorer's own hidden layer,
        # predicting the searched decision's per-option mean simulation value minus the
        # root value. TRAINING ONLY -- no bundle-facing forward calls it, so every
        # inference path (policy_value / policy_value_context / policy_value_tokens) keeps
        # its exact signature and outputs. ADDITIVE: a checkpoint saved before this head
        # existed has no q_head.* keys, and load_state_dict below fills them in from the
        # fresh init so old checkpoints still load strictly.
        self.q_head = nn.Linear(config.d_model, 1) if config.option_feature_dim else None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Tolerate PRE-q_head checkpoints: fill the missing q_head.* entries from this
        module's freshly initialized head (the head then trains from scratch, exactly like
        a newly added aux head does). Any other key disagreement is still rejected."""
        if self.q_head is not None \
                and not any(key.startswith("q_head.") for key in state_dict):
            own = super().state_dict()
            state_dict = dict(state_dict)
            for name in ("q_head.weight", "q_head.bias"):
                state_dict[name] = own[name]
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def option_q(self, context, option_features, option_mask=None):
        """TRAINER-ONLY Q prediction, one scalar per option row [B, O].

        Reads the SAME hidden the policy logit is projected from -- policy_score is
        Linear -> GELU -> Linear, so this is that GELU output with a different final
        projection. Recomputed here rather than returned from the forwards so the shipped
        entry points stay byte-identical."""
        broadcast = context.unsqueeze(1).expand(-1, option_features.shape[1], -1)
        hidden = self.policy_score[1](
            self.policy_score[0](torch.cat([broadcast, option_features], dim=-1)))
        q = self.q_head(hidden).squeeze(-1)
        return q if option_mask is None else q.masked_fill(~option_mask, 0.0)

    def _card_input(self, token_features, card_ids):
        """Token features, with the learned identity embedding concatenated when this model
        has one. `card_ids` [B, T] indexes it (0 = no card, e.g. history rows)."""
        if self.card_embedding is None:
            return token_features
        if card_ids is None:
            raise ValueError("this model was built with card_vocab > 0 and needs card_ids "
                             "(the v2 encoders emit them alongside token_features)")
        return torch.cat([token_features, self.card_embedding(card_ids)], dim=-1)

    def _encode(self, token_features, owner_ids, zone_ids, padding_mask, global_features,
                card_ids=None):
        """Run the shared trunk; return the global/CLS token's output [B, d_model] -- the board
        context after attention over every token."""
        batch_size = token_features.shape[0]
        device = token_features.device

        # Card tokens: project, then add owner + zone segment embeddings.
        cards = self.card_projection(self._card_input(token_features, card_ids))
        cards = cards + self.owner_embedding(owner_ids) + self.zone_embedding(zone_ids)

        # Global token: its own projection + the reserved neutral/global segments.
        global_token = self.global_projection(global_features).unsqueeze(1)        # [B, 1, d]
        owner = torch.full((batch_size, 1), OWNER_NEUTRAL, device=device)
        zone = torch.full((batch_size, 1), ZONE_GLOBAL, device=device)
        global_token = global_token + self.owner_embedding(owner) + self.zone_embedding(zone)

        # Prepend the global token (never padded) and run full self-attention.
        sequence = torch.cat([global_token, cards], dim=1)                         # [B, 1+T, d]
        not_padded = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([not_padded, padding_mask], dim=1)            # True = ignore
        encoded = self.encoder(sequence, src_key_padding_mask=key_padding_mask)
        return encoded[:, 0]                                                        # [B, d_model]

    def forward(self, token_features, owner_ids, zone_ids, padding_mask, global_features,
                card_ids=None):
        """Value only (the value-net path) -- scalar in [-1, 1] per state."""
        context = self._encode(token_features, owner_ids, zone_ids, padding_mask,
                               global_features, card_ids)
        return torch.tanh(self.value_head(context)).squeeze(-1)

    def forward_tokens(self, token_features, owner_ids, zone_ids, padding_mask, global_features,
                       card_ids=None):
        """Like forward() but also exposes the per-CARD-token embeddings, for auxiliary per-token
        heads on the shared trunk. Returns (value [B], context [B, d_model], token_embeddings
        [B, T, d_model]). `context` is the CLS output the value head reads (same tensor forward()
        uses); `token_embeddings` are the encoded card tokens in encode_game's emission order
        (board tokens lead, so the first len(board_serials) are the in-play Pokemon). Duplicates
        _encode's body deliberately so forward()/_encode()/policy_value() stay byte-identical and
        the trunk state_dict is unchanged (the aux heads live outside this module)."""
        batch_size = token_features.shape[0]
        device = token_features.device
        cards = self.card_projection(self._card_input(token_features, card_ids))
        cards = cards + self.owner_embedding(owner_ids) + self.zone_embedding(zone_ids)
        global_token = self.global_projection(global_features).unsqueeze(1)
        owner = torch.full((batch_size, 1), OWNER_NEUTRAL, device=device)
        zone = torch.full((batch_size, 1), ZONE_GLOBAL, device=device)
        global_token = global_token + self.owner_embedding(owner) + self.zone_embedding(zone)
        sequence = torch.cat([global_token, cards], dim=1)
        not_padded = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([not_padded, padding_mask], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=key_padding_mask)
        context = encoded[:, 0]                       # CLS -> value + belief heads
        token_embeddings = encoded[:, 1:]             # per card token -> threat + ko heads
        value = torch.tanh(self.value_head(context)).squeeze(-1)
        return value, context, token_embeddings

    def policy_value_context(self, token_features, owner_ids, zone_ids, padding_mask,
                             global_features, option_features, option_mask, card_ids=None):
        """policy_value plus the CLS context [B, d_model], for auxiliary heads on the shared
        trunk during training (same convention as forward_tokens: the aux heads live OUTSIDE
        this module so every trunk state_dict stays loadable everywhere)."""
        context = self._encode(token_features, owner_ids, zone_ids, padding_mask,
                               global_features, card_ids)
        value = torch.tanh(self.value_head(context)).squeeze(-1)
        broadcast = context.unsqueeze(1).expand(-1, option_features.shape[1], -1)
        logits = self.policy_score(torch.cat([broadcast, option_features], dim=-1)).squeeze(-1)
        return logits.masked_fill(~option_mask, -1e9), value, context

    def policy_value_tokens(self, token_features, owner_ids, zone_ids, padding_mask,
                            global_features, option_features, option_mask, card_ids=None):
        """policy_value plus the CLS context AND the per-card token embeddings, for per-token
        auxiliary heads (KO clocks) alongside the CLS heads during training. Same convention
        as forward_tokens / policy_value_context: aux heads live OUTSIDE this module, the
        trunk state_dict stays loadable everywhere. Returns (logits, value, context,
        token_embeddings [B, T, d_model] in encode_game's emission order -- board tokens
        lead, so the first len(board_serials) are the in-play Pokemon)."""
        batch_size = token_features.shape[0]
        device = token_features.device
        cards = self.card_projection(self._card_input(token_features, card_ids))
        cards = cards + self.owner_embedding(owner_ids) + self.zone_embedding(zone_ids)
        global_token = self.global_projection(global_features).unsqueeze(1)
        owner = torch.full((batch_size, 1), OWNER_NEUTRAL, device=device)
        zone = torch.full((batch_size, 1), ZONE_GLOBAL, device=device)
        global_token = global_token + self.owner_embedding(owner) + self.zone_embedding(zone)
        sequence = torch.cat([global_token, cards], dim=1)
        not_padded = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([not_padded, padding_mask], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=key_padding_mask)
        context = encoded[:, 0]
        token_embeddings = encoded[:, 1:]
        value = torch.tanh(self.value_head(context)).squeeze(-1)
        broadcast = context.unsqueeze(1).expand(-1, option_features.shape[1], -1)
        logits = self.policy_score(torch.cat([broadcast, option_features], dim=-1)).squeeze(-1)
        return logits.masked_fill(~option_mask, -1e9), value, context, token_embeddings

    def policy_value(self, token_features, owner_ids, zone_ids, padding_mask, global_features,
                     option_features, option_mask, card_ids=None):
        """The actor-critic head: one logit per legal option + the state value, sharing the trunk.
        option_features [B, O, opt], option_mask [B, O] (True = real option). Padded options get
        a large-negative logit so they vanish under softmax. Returns (logits [B, O], value [B])."""
        context = self._encode(token_features, owner_ids, zone_ids, padding_mask,
                               global_features, card_ids)
        value = torch.tanh(self.value_head(context)).squeeze(-1)                    # [B]
        broadcast = context.unsqueeze(1).expand(-1, option_features.shape[1], -1)   # [B, O, d]
        logits = self.policy_score(torch.cat([broadcast, option_features], dim=-1)).squeeze(-1)
        return logits.masked_fill(~option_mask, -1e9), value


if __name__ == "__main__":
    config = GameStateTransformerConfig(token_feature_dim=114, global_feature_dim=22)
    model = GameStateTransformer(config)

    batch, tokens = 4, 30
    token_features = torch.randn(batch, tokens, config.token_feature_dim)
    owner_ids = torch.randint(OWNER_ME, OWNER_OPPONENT + 1, (batch, tokens))
    zone_ids = torch.randint(ZONE_ACTIVE, NUM_ZONES, (batch, tokens))
    padding_mask = torch.zeros(batch, tokens, dtype=torch.bool)
    padding_mask[:, 20:] = True
    global_features = torch.randn(batch, config.global_feature_dim)

    value = model(token_features, owner_ids, zone_ids, padding_mask, global_features)
    print(f"value shape {tuple(value.shape)}  range [{value.min():.3f}, {value.max():.3f}]")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
