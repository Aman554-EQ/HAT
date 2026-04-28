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
# Role in pipeline:  encoded_x → HierarchicalContextEncoder → (ctx_encoded, ctx_out)
#                                                                     ↓
#                                                              DualMemoryUnit
#
# Why it exists:  HAT passed raw encoded_x (plain short-window encoder output)
# as the context bridge into the memory unit.  This module enriches that bridge
# via two stages:
#   Stage 1 (ctx_encoder)  — instance-level self-attention over encoded_x,
#                            producing richer per-frame representations.
#   Stage 2 (ctx_decoder)  — cross-attention: learnable ctx_token queries
#                            distil the window into compact context tokens.
#
# Design follows the HAT pattern exactly:
#   HAT added history_encoder_block1 (cross-attn) and block2 (cross-attn)
#   inside HistoryUnit.  We add ctx_encoder (self-attn) and ctx_decoder
#   (cross-attn) inside HierarchicalContextEncoder — same building blocks,
#   applied to the short-window side.
# ---------------------------------------------------------------------------
class HierarchicalContextEncoder(torch.nn.Module):
    def __init__(self, opt):
        super(HierarchicalContextEncoder, self).__init__()
        n_embedding_dim   = opt["hidden_dim"]
        self.n_ctx_tokens = opt.get("ctx_tokens", 8)
        n_ctx_enc_head    = 4
        n_ctx_enc_layer   = 2   # instance-level self-attention layers
        n_ctx_dec_head    = 4
        n_ctx_dec_layer   = 2   # cross-attention layers → compact tokens
        dropout           = 0.3

        # Stage 1: local instance-level self-attention over the short window
        self.ctx_encoder = nn.TransformerEncoder(
                                nn.TransformerEncoderLayer(d_model=n_embedding_dim,
                                                           nhead=n_ctx_enc_head,
                                                           dropout=dropout,
                                                           activation='gelu'),
                                n_ctx_enc_layer,
                                nn.LayerNorm(n_embedding_dim))

        # Stage 2: compact context tokens extracted via cross-attention
        self.ctx_decoder = nn.TransformerDecoder(
                                nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                           nhead=n_ctx_dec_head,
                                                           dropout=dropout,
                                                           activation='gelu'),
                                n_ctx_dec_layer,
                                nn.LayerNorm(n_embedding_dim))

        # Learnable instance-level context queries (mirrors history_token in HAT)
        self.ctx_token = nn.Parameter(torch.zeros(self.n_ctx_tokens, 1, n_embedding_dim))

        self.ctx_norm  = nn.LayerNorm(n_embedding_dim)
        self.ctx_drop  = nn.Dropout(0.1)

    def forward(self, encoded_x):
        # encoded_x : [short_window_size, B, D]

        # Stage 1 — instance-level self-attention
        ctx_encoded = self.ctx_encoder(encoded_x)                           # [16, B, D]
        ctx_encoded = self.ctx_norm(ctx_encoded + self.ctx_drop(encoded_x)) # residual + norm

        # Stage 2 — distil compact context tokens
        ctx_token = self.ctx_token.expand(-1, encoded_x.shape[1], -1)       # [ctx_tokens, B, D]
        ctx_out   = self.ctx_decoder(ctx_token, ctx_encoded)                 # [ctx_tokens, B, D]

        return ctx_encoded, ctx_out


# ---------------------------------------------------------------------------
# HAT+ Module 2: DualMemoryUnit  (replaces HistoryUnit)
# ---------------------------------------------------------------------------
# Role in pipeline:  long_x + (ctx_encoded, ctx_out)
#                        → DualMemoryUnit → (fused_mem, snip_cls)
#                                                  ↓
#                                   history_anchor_decoder_block1 in MYNET
#
# Relationship to HistoryUnit:
#   • long_mem_encoder  ←→  history_encoder_block1  (preserved exactly)
#   • short_mem_encoder  — NEW stream: ctx_out as key/value
#   • memory_fusion      ←→  history_encoder_block2  (same depth/heads,
#                            but now queries long-term memory using
#                            structured short-term tokens instead of raw
#                            encoded_x sequence)
#   • snip_head / snip_classifier — identical to HAT
#   • history_token, norm2, dropout2  — identical to HAT
#
# Key innovation:  HAT's block2 blended history tokens with a raw 16-frame
# sequence (encoded_x).  DualMemoryUnit first compresses the current context
# into compact short_mem tokens (via HierarchicalContextEncoder + this
# short_mem_encoder), then fuses long-term memory with that structured
# summary.  The fusion is more discriminative and noise-tolerant.
# ---------------------------------------------------------------------------
class DualMemoryUnit(torch.nn.Module):
    def __init__(self, opt):
        super(DualMemoryUnit, self).__init__()
        self.n_feature         = opt["feat_dim"]
        n_class                = opt["num_of_class"]
        n_embedding_dim        = opt["hidden_dim"]
        # Long-term memory — same architecture as HAT's history_encoder_block1
        n_hist_dec_head        = 4
        n_hist_dec_layer       = 5
        # Short-term memory — new stream, mirrors long-term structure
        n_short_dec_head       = 4
        n_short_dec_layer      = 2
        # Memory fusion — replaces HAT's history_encoder_block2
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

        # Positional encoding for the long historical sequence (same as HAT)
        self.history_positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)

        # Long-term memory encoder: history_token queries attend to long_x
        # ← HAT's history_encoder_block1, preserved exactly
        self.long_mem_encoder = nn.TransformerDecoder(
                                    nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                               nhead=n_hist_dec_head,
                                                               dropout=dropout,
                                                               activation='gelu'),
                                    n_hist_dec_layer,
                                    nn.LayerNorm(n_embedding_dim))

        # Short-term memory encoder: short_mem_token queries attend to ctx_out
        # ← NEW — mirrors long_mem_encoder in structure
        self.short_mem_encoder = nn.TransformerDecoder(
                                    nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                               nhead=n_short_dec_head,
                                                               dropout=dropout,
                                                               activation='gelu'),
                                    n_short_dec_layer,
                                    nn.LayerNorm(n_embedding_dim))

        # Memory fusion: long_mem tokens attend to short_mem tokens
        # ← replaces HAT's history_encoder_block2
        self.memory_fusion = nn.TransformerDecoder(
                                    nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                               nhead=n_fusion_dec_head,
                                                               dropout=dropout,
                                                               activation='gelu'),
                                    n_fusion_dec_layer,
                                    nn.LayerNorm(n_embedding_dim))

        # Snippet classification head — identical to HAT
        self.snip_head = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim // 4), nn.ReLU())
        self.snip_classifier = nn.Sequential(
            nn.Linear(self.history_tokens * n_embedding_dim // 4,
                      (self.history_tokens * n_embedding_dim // 4) // 4),
            nn.ReLU(),
            nn.Linear((self.history_tokens * n_embedding_dim // 4) // 4, n_class))

        # Learnable query tokens
        # history_token — long-term queries, same as HAT
        self.history_token   = nn.Parameter(torch.zeros(self.history_tokens,   1, n_embedding_dim))
        # short_mem_token — short-term queries, NEW
        self.short_mem_token = nn.Parameter(torch.zeros(self.short_mem_tokens, 1, n_embedding_dim))

        self.norm2    = nn.LayerNorm(n_embedding_dim)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, long_x, ctx_encoded, ctx_out):
        # long_x      : [48, B, D]   — historical frames (before short window)
        # ctx_encoded : [16, B, D]   — enriched short-window features (Stage 1 output)
        # ctx_out     : [ctx_tokens, B, D] — compact context tokens (Stage 2 output)

        ## Long-term Memory  (HAT's block1 — preserved)
        hist_pe_x     = self.history_positional_encoding(long_x)
        history_token = self.history_token.expand(-1, hist_pe_x.shape[1], -1)
        long_mem      = self.long_mem_encoder(history_token, hist_pe_x)  # [16, B, D]

        ## Short-term Memory  (NEW — ctx_out as key/value source)
        short_mem_token = self.short_mem_token.expand(-1, ctx_out.shape[1], -1)
        short_mem       = self.short_mem_encoder(short_mem_token, ctx_out) # [short_mem_tokens, B, D]

        ## Memory Fusion  (replaces HAT's block2 — long queries short)
        fused_mem = self.memory_fusion(long_mem, short_mem)               # [16, B, D]
        fused_mem = fused_mem + self.dropout2(long_mem)                   # residual (HAT style)
        fused_mem = self.norm2(fused_mem)

        ## Snippet Classification Head  (identical to HAT — uses long_mem)
        # Note: snip_cls is computed from long_mem (pre-fusion), same as HAT
        # computing from hist_encoded_x_1.  This keeps the supervision signal
        # independent of the short-term stream, enabling clean ablation.
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

        # HAT+ modules — replace self.history_unit = HistoryUnit(opt)
        # Pipeline:  encoded_x → context_encoder → dual_memory_unit → anchor refinement
        self.context_encoder  = HierarchicalContextEncoder(opt)
        self.dual_memory_unit = DualMemoryUnit(opt)

        # History-driven anchor refinement — unchanged from HAT
        self.history_anchor_decoder_block1 = nn.TransformerDecoder(
                            nn.TransformerDecoderLayer(d_model=n_embedding_dim,
                                                       nhead=n_comb_dec_head,
                                                       dropout=dropout,
                                                       activation='gelu'),
                            n_comb_dec_layer,
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

        self.relu      = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # inputs: [B, T, feat_dim]  where T = segment_size (default 64)

        # Feature projection — unchanged from HAT
        base_x_rgb  = self.feature_reduction_rgb(inputs[:, :, :self.n_feature // 2].float())
        base_x_flow = self.feature_reduction_flow(inputs[:, :, self.n_feature // 2:].float())
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)

        base_x = base_x.permute([1, 0, 2])  # [T, B, D]

        # Temporal split — unchanged from HAT
        short_x = base_x[-self.short_window_size:]   # [16, B, D] — current window
        long_x  = base_x[:-self.short_window_size]   # [48, B, D] — historical context

        ## Anchor Feature Generator — unchanged from HAT
        pe_x          = self.positional_encoding(short_x)
        encoded_x     = self.encoder(pe_x)                                       # [16, B, D]
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)   # [n_anchors, B, D]
        decoded_x     = self.decoder(decoder_token, encoded_x)                   # [n_anchors, B, D]

        ## HAT+: Hierarchical Context Encoding
        # encoded_x → richer per-frame features (ctx_encoded) +
        #              compact instance-level tokens (ctx_out)
        ctx_encoded, ctx_out = self.context_encoder(encoded_x)

        ## HAT+: Dual Memory Unit  (replaces history_unit call)
        # long_x + ctx_encoded + ctx_out → fused memory tokens + snippet prediction
        hist_encoded_x, snip_cls = self.dual_memory_unit(long_x, ctx_encoded, ctx_out)

        ## History-Driven Anchor Refinement — unchanged from HAT
        decoded_anchor_feat = self.history_anchor_decoder_block1(decoded_x, hist_encoded_x)
        decoded_anchor_feat = decoded_anchor_feat + self.dropout1(decoded_x)
        decoded_anchor_feat = self.norm1(decoded_anchor_feat)
        decoded_anchor_feat = decoded_anchor_feat.permute([1, 0, 2])  # [B, n_anchors, D]

        # Prediction Module — unchanged from HAT
        anc_cls = self.classifier(decoded_anchor_feat)
        anc_reg = self.regressor(decoded_anchor_feat)

        # Return signature identical to HAT: (anc_cls, anc_reg, snip_cls)
        return anc_cls, anc_reg, snip_cls


class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class        = opt["num_of_class"] - 1
        n_seglen       = opt["segment_size"]
        n_embedding_dim = 2 * n_seglen
        dropout        = 0.3
        self.best_loss = 1000000
        self.best_map  = 0

        self.mlp1    = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2    = nn.Linear(n_embedding_dim, 1)
        self.norm    = nn.InstanceNorm1d(n_class)
        self.relu    = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # inputs: [B, seq_len, n_class]
        base_x = inputs.permute([0, 2, 1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        return x
