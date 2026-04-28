import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import normalize


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 750):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# ---------------------------------------------------------------------------
# HAT+ Module 1: HierarchicalContextEncoder
# ---------------------------------------------------------------------------
# Enriches the short-window encoder output via two stages:
#   Stage 1 (ctx_encoder)  — instance-level self-attention over encoded_x
#   Stage 2 (ctx_decoder)  — compact context tokens via cross-attention
# Also produces ctx_cls: a context-level classification prediction used
# as direct auxiliary supervision (same snip_label, same focal loss).
# This provides a direct gradient signal to the context encoder every step,
# ensuring it learns to represent current activity — not just pass gradients
# through the memory chain.
# ---------------------------------------------------------------------------
class HierarchicalContextEncoder(torch.nn.Module):
    def __init__(self, opt):
        super(HierarchicalContextEncoder, self).__init__()
        n_embedding_dim   = opt["hidden_dim"]
        n_class           = opt["num_of_class"]
        self.n_ctx_tokens = opt.get("ctx_tokens", 8)
        n_ctx_enc_head    = 4
        n_ctx_enc_layer   = 2
        n_ctx_dec_head    = 4
        n_ctx_dec_layer   = 2
        dropout           = 0.3

        # Stage 1: local instance-level self-attention
        self.ctx_encoder = nn.TransformerEncoder(
                                nn.TransformerEncoderLayer(d_model=n_embedding_dim,
                                                           nhead=n_ctx_enc_head,
                                                           dropout=dropout,
                                                           activation='gelu'),
                                n_ctx_enc_layer,
                                nn.LayerNorm(n_embedding_dim))

        # Stage 2: compact context tokens via cross-attention
        self.ctx_decoder = nn.TransformerDecoder(
                                nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                           nhead=n_ctx_dec_head,
                                                           dropout=dropout,
                                                           activation='gelu'),
                                n_ctx_dec_layer,
                                nn.LayerNorm(n_embedding_dim))

        # Learnable context queries (mirrors history_token in HAT)
        self.ctx_token = nn.Parameter(torch.zeros(self.n_ctx_tokens, 1, n_embedding_dim))

        self.ctx_norm  = nn.LayerNorm(n_embedding_dim)
        self.ctx_drop  = nn.Dropout(0.1)

        # Context supervision head — provides direct gradient to this module.
        # Uses the same snip_label as the history-level snippet head.
        # Follows HAT's snip_head pattern exactly, scaled to ctx_tokens.
        self.ctx_head = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim // 4), nn.ReLU())
        self.ctx_classifier = nn.Sequential(
            nn.Linear(self.n_ctx_tokens * n_embedding_dim // 4,
                      (self.n_ctx_tokens * n_embedding_dim // 4) // 4),
            nn.ReLU(),
            nn.Linear((self.n_ctx_tokens * n_embedding_dim // 4) // 4, n_class))

    def forward(self, encoded_x):
        # encoded_x : [short_window_size, B, D]

        # Stage 1 — instance-level self-attention + residual
        ctx_encoded = self.ctx_encoder(encoded_x)                            # [16, B, D]
        ctx_encoded = self.ctx_norm(ctx_encoded + self.ctx_drop(encoded_x))  # residual + norm

        # Stage 2 — compact context tokens
        ctx_token = self.ctx_token.expand(-1, encoded_x.shape[1], -1)        # [ctx_tokens, B, D]
        ctx_out   = self.ctx_decoder(ctx_token, ctx_encoded)                  # [ctx_tokens, B, D]

        # Context classification (direct supervision, HAT snip_head style)
        ctx_feat = self.ctx_head(ctx_out)                                     # [ctx_tokens, B, D//4]
        ctx_feat = torch.flatten(ctx_feat.permute(1, 0, 2), start_dim=1)     # [B, ctx_tokens*D//4]
        ctx_cls  = self.ctx_classifier(ctx_feat)                              # [B, n_class]

        return ctx_encoded, ctx_out, ctx_cls


# ---------------------------------------------------------------------------
# HAT+ Module 2: DualMemoryUnit  (replaces HistoryUnit)
# ---------------------------------------------------------------------------
# Three sub-components:
#   long_mem_encoder — HAT's history_encoder_block1, preserved exactly
#   short_mem_encoder — NEW: compresses ctx_out into short-term tokens
#   memory_fusion     — replaces block2; additionally gated by short-term
#                       summary so the network can selectively weight
#                       which history tokens are relevant right now
# ---------------------------------------------------------------------------
class DualMemoryUnit(torch.nn.Module):
    def __init__(self, opt):
        super(DualMemoryUnit, self).__init__()
        self.n_feature         = opt["feat_dim"]
        n_class                = opt["num_of_class"]
        n_embedding_dim        = opt["hidden_dim"]
        n_hist_dec_head        = 4
        n_hist_dec_layer       = 5    # same as HAT's history_encoder_block1
        n_short_dec_head       = 4
        n_short_dec_layer      = 2
        n_fusion_dec_head      = 4
        n_fusion_dec_layer     = 2
        self.anchors           = opt["anchors"]
        self.history_tokens    = 16
        self.short_mem_tokens  = opt.get("short_mem_tokens", 8)
        self.short_window_size = 16
        self.anchors_stride    = []
        dropout                = 0.3
        self.best_loss         = 1000000
        self.best_map          = 0

        # Positional encoding for long_x (same as HAT)
        self.history_positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)

        # Long-term memory: history_token queries × long_x  ← HAT block1, preserved
        self.long_mem_encoder = nn.TransformerDecoder(
                                    nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                               nhead=n_hist_dec_head,
                                                               dropout=dropout,
                                                               activation='gelu'),
                                    n_hist_dec_layer,
                                    nn.LayerNorm(n_embedding_dim))

        # Short-term memory: short_mem_token queries × ctx_out  ← NEW
        self.short_mem_encoder = nn.TransformerDecoder(
                                    nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                               nhead=n_short_dec_head,
                                                               dropout=dropout,
                                                               activation='gelu'),
                                    n_short_dec_layer,
                                    nn.LayerNorm(n_embedding_dim))

        # Memory fusion: long_mem queries × short_mem  ← replaces HAT block2
        self.memory_fusion = nn.TransformerDecoder(
                                    nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                               nhead=n_fusion_dec_head,
                                                               dropout=dropout,
                                                               activation='gelu'),
                                    n_fusion_dec_layer,
                                    nn.LayerNorm(n_embedding_dim))

        # Gated memory integration: scalar gate per history token, conditioned
        # on the short-term context summary.  Allows the network to learn
        # which historical tokens are relevant for the current moment.
        # Output is element-wise multiplied onto long_mem before fusion.
        self.mem_gate = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(n_embedding_dim // 4, n_embedding_dim),
            nn.Sigmoid())

        # Snippet classification head — identical to HAT
        self.snip_head = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim // 4), nn.ReLU())
        self.snip_classifier = nn.Sequential(
            nn.Linear(self.history_tokens * n_embedding_dim // 4,
                      (self.history_tokens * n_embedding_dim // 4) // 4),
            nn.ReLU(),
            nn.Linear((self.history_tokens * n_embedding_dim // 4) // 4, n_class))

        # Learnable query tokens
        self.history_token   = nn.Parameter(torch.zeros(self.history_tokens,   1, n_embedding_dim))
        self.short_mem_token = nn.Parameter(torch.zeros(self.short_mem_tokens, 1, n_embedding_dim))

        self.norm2    = nn.LayerNorm(n_embedding_dim)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, long_x, ctx_encoded, ctx_out):
        # long_x      : [48, B, D]
        # ctx_encoded : [16, B, D]  — enriched short-window (Stage 1)
        # ctx_out     : [ctx_tokens, B, D]  — compact context tokens (Stage 2)

        ## Long-term Memory  (HAT block1 — preserved)
        hist_pe_x     = self.history_positional_encoding(long_x)
        history_token = self.history_token.expand(-1, hist_pe_x.shape[1], -1)
        long_mem      = self.long_mem_encoder(history_token, hist_pe_x)    # [16, B, D]

        ## Short-term Memory  (NEW)
        short_mem_token = self.short_mem_token.expand(-1, ctx_out.shape[1], -1)
        short_mem       = self.short_mem_encoder(short_mem_token, ctx_out) # [8, B, D]

        ## Gated scaling of long_mem before fusion
        # Gate is conditioned on short_mem mean — gives each history token
        # a data-driven relevance weight relative to the current context.
        short_summary  = short_mem.mean(dim=0)           # [B, D]
        gate           = self.mem_gate(short_summary)    # [B, D]
        gate           = gate.unsqueeze(0)               # [1, B, D]
        long_mem_gated = long_mem * gate                 # [16, B, D]

        ## Memory Fusion  (replaces HAT block2 — long_gated queries short)
        fused_mem = self.memory_fusion(long_mem_gated, short_mem)  # [16, B, D]
        fused_mem = fused_mem + self.dropout2(long_mem)            # residual from ungated (HAT style)
        fused_mem = self.norm2(fused_mem)

        ## Snippet Classification Head  (identical to HAT — uses pre-gate long_mem)
        snippet_feat = self.snip_head(long_mem)
        snippet_feat = torch.flatten(snippet_feat.permute(1, 0, 2), start_dim=1)
        snip_cls     = self.snip_classifier(snippet_feat)

        return fused_mem, snip_cls


class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature      = opt["feat_dim"]
        n_class             = opt["num_of_class"]
        n_embedding_dim     = opt["hidden_dim"]
        n_enc_layer         = opt["enc_layer"]
        n_enc_head          = opt["enc_head"]
        n_dec_layer         = opt["dec_layer"]
        n_dec_head          = opt["dec_head"]
        n_comb_dec_head     = 4
        n_comb_dec_layer    = 5
        n_seglen            = opt["segment_size"]
        self.anchors        = opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride = []
        dropout             = 0.3
        self.best_loss      = 1000000
        self.best_map       = 0

        # Feature projection — unchanged from HAT
        self.feature_reduction_rgb  = nn.Linear(self.n_feature // 2, n_embedding_dim // 2)
        self.feature_reduction_flow = nn.Linear(self.n_feature // 2, n_embedding_dim // 2)

        # Positional encoding + short-window encoder + anchor decoder — unchanged from HAT
        self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)

        self.encoder = nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(d_model=n_embedding_dim,
                                                       nhead=n_enc_head,
                                                       dropout=dropout,
                                                       activation='gelu'),
                            n_enc_layer,
                            nn.LayerNorm(n_embedding_dim))

        self.decoder = nn.TransformerDecoder(
                            nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                       nhead=n_dec_head,
                                                       dropout=dropout,
                                                       activation='gelu'),
                            n_dec_layer,
                            nn.LayerNorm(n_embedding_dim))

        # HAT+ modules
        self.context_encoder  = HierarchicalContextEncoder(opt)
        self.dual_memory_unit = DualMemoryUnit(opt)

        # Stage 1 anchor refinement: history-driven  ← unchanged from HAT
        self.history_anchor_decoder_block1 = nn.TransformerDecoder(
                            nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                       nhead=n_comb_dec_head,
                                                       dropout=dropout,
                                                       activation='gelu'),
                            n_comb_dec_layer,
                            nn.LayerNorm(n_embedding_dim))

        # Stage 2 anchor refinement: context-driven  ← NEW (HAT+)
        # After history-driven coarse refinement, anchors attend to enriched
        # short-window context for a second, fine-grained pass.
        # Lightweight: 2 layers vs 5 in Stage 1.
        self.context_anchor_decoder_block = nn.TransformerDecoder(
                            nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                       nhead=n_comb_dec_head,
                                                       dropout=dropout,
                                                       activation='gelu'),
                            2,
                            nn.LayerNorm(n_embedding_dim))

        # Classification and regression heads — unchanged from HAT
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(),
            nn.Linear(n_embedding_dim, n_class))
        self.regressor = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(),
            nn.Linear(n_embedding_dim, 2))

        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))

        self.norm1    = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.norm3    = nn.LayerNorm(n_embedding_dim)  # Stage 2 norm
        self.dropout3 = nn.Dropout(0.1)               # Stage 2 dropout

        self.relu      = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # Feature projection — unchanged
        base_x_rgb  = self.feature_reduction_rgb(inputs[:, :, :self.n_feature // 2].float())
        base_x_flow = self.feature_reduction_flow(inputs[:, :, self.n_feature // 2:].float())
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        base_x = base_x.permute([1, 0, 2])  # [T, B, D]

        # Temporal split — unchanged
        short_x = base_x[-self.short_window_size:]   # [16, B, D]
        long_x  = base_x[:-self.short_window_size]   # [48, B, D]

        ## Anchor Feature Generator — unchanged
        pe_x          = self.positional_encoding(short_x)
        encoded_x     = self.encoder(pe_x)
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)
        decoded_x     = self.decoder(decoder_token, encoded_x)

        ## HAT+: Hierarchical Context Encoding
        # Returns enriched short features, compact tokens, AND context cls prediction
        ctx_encoded, ctx_out, ctx_cls = self.context_encoder(encoded_x)

        ## HAT+: Dual Memory Unit
        hist_encoded_x, snip_cls = self.dual_memory_unit(long_x, ctx_encoded, ctx_out)

        ## Stage 1: History-Driven Anchor Refinement — unchanged from HAT
        after_history = self.history_anchor_decoder_block1(decoded_x, hist_encoded_x)
        after_history = after_history + self.dropout1(decoded_x)
        after_history = self.norm1(after_history)

        ## Stage 2: Context-Driven Anchor Refinement — NEW (HAT+)
        # Anchors cross-attend to the enriched short-window features.
        # Gives each anchor a fine-grained local-context signal after the
        # coarse history-driven pass, enabling hierarchical refinement.
        decoded_anchor_feat = self.context_anchor_decoder_block(after_history, ctx_encoded)
        decoded_anchor_feat = decoded_anchor_feat + self.dropout3(after_history)  # residual from Stage 1
        decoded_anchor_feat = self.norm3(decoded_anchor_feat)
        decoded_anchor_feat = decoded_anchor_feat.permute([1, 0, 2])  # [B, n_anchors, D]

        # Prediction Module — unchanged
        anc_cls = self.classifier(decoded_anchor_feat)
        anc_reg = self.regressor(decoded_anchor_feat)

        # Return: (anc_cls, anc_reg, snip_cls, ctx_cls)
        # ctx_cls is the new context supervision output
        return anc_cls, anc_reg, snip_cls, ctx_cls


class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class         = opt["num_of_class"] - 1
        n_seglen        = opt["segment_size"]
        n_embedding_dim = 2 * n_seglen
        dropout         = 0.3
        self.best_loss  = 1000000
        self.best_map   = 0

        self.mlp1    = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2    = nn.Linear(n_embedding_dim, 1)
        self.norm    = nn.InstanceNorm1d(n_class)
        self.relu    = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        base_x = inputs.permute([0, 2, 1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        return x
