"""TFT-style network: gated residual blocks + LSTM encoder-decoder +
interpretable attention + direct multi-quantile head.

This is TFT-*style*, not the full paper: it keeps the load-bearing ideas
(input routing static/known/observed, an LSTM local layer, attention over
the encoded past, direct multi-horizon quantile output) and drops the
gadgets that don't change the lesson (per-variable selection networks are
replaced by projected input blocks; interpretable attention is approximated
by head-averaged standard attention). The doc says so plainly.

Key contrast with DeepAR (Phase 10):
  DeepAR  : recurrence + a likelihood + ancestral SAMPLING for quantiles
  TFT     : attention + DIRECT quantile outputs (all 28 days x 7 quantiles
            in one forward pass) trained on pinball loss — no sampling, no
            error feedback, and attention weights you can read.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GRN(nn.Module):
    """Gated Residual Network — the TFT workhorse block.

    GLU-gated non-linearity with a residual skip and layer norm, optionally
    conditioned on a static context vector. The gate lets the network route
    *around* the block when a transform isn't needed — cheap adaptivity.
    """

    def __init__(self, in_dim, hid_dim, out_dim, ctx_dim=None, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.ctx = nn.Linear(ctx_dim, hid_dim, bias=False) if ctx_dim else None
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.glu = nn.Linear(out_dim, out_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.skip = None if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, a, context=None):
        h = self.fc1(a)
        if self.ctx is not None and context is not None:
            c = context if context.dim() == a.dim() else context.unsqueeze(1)
            h = h + self.ctx(c)
        h = self.drop(self.fc2(F.elu(h)))
        val, gate = self.glu(h).chunk(2, dim=-1)
        skip = a if self.skip is None else self.skip(a)
        return self.norm(skip + val * torch.sigmoid(gate))


class TFTNet(nn.Module):
    def __init__(self, n_items, n_depts, n_stores, cov_dim, context_len, horizon,
                 quantiles, hidden=64, heads=4, dropout=0.1, embed_dim=16):
        super().__init__()
        self.context_len, self.horizon = context_len, horizon
        self.register_buffer("quantiles", torch.tensor(quantiles), persistent=False)

        self.emb_item = nn.Embedding(n_items, embed_dim)
        self.emb_dept = nn.Embedding(n_depts, 4)
        self.emb_store = nn.Embedding(n_stores, 4)
        static_dim = embed_dim + 4 + 4
        # static context conditions the temporal blocks (paper: 4 context vectors;
        # we share one for compactness)
        self.static_grn = GRN(static_dim, hidden, hidden, dropout=dropout)

        # input routing: observed (sales+covs) feeds the encoder; known (covs)
        # feeds the decoder — the future can't see sales
        self.enc_proj = GRN(1 + cov_dim, hidden, hidden, ctx_dim=hidden, dropout=dropout)
        self.dec_proj = GRN(cov_dim, hidden, hidden, ctx_dim=hidden, dropout=dropout)

        self.lstm_enc = nn.LSTM(hidden, hidden, batch_first=True)
        self.lstm_dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.gate_lstm = GRN(hidden, hidden, hidden, dropout=dropout)

        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.attn_grn = GRN(hidden, hidden, hidden, dropout=dropout)
        self.pos_grn = GRN(hidden, hidden, hidden, dropout=dropout)
        self.head = nn.Linear(hidden, len(quantiles))

    def _attn_mask(self, C: int, H: int, device) -> torch.Tensor:
        """[H, C+H] bool, False = allowed. Decoder query i (abs time C+i) may
        attend to keys j <= C+i: all of the encoder + decoder steps up to i
        (causal). Built per-call so a shorter prediction horizon still works."""
        j = torch.arange(C + H, device=device).unsqueeze(0)   # [1, C+H]
        cutoff = C + torch.arange(H, device=device).unsqueeze(1)  # [H, 1]
        return j > cutoff  # True where disallowed

    def _static(self, cats):
        emb = torch.cat([self.emb_item(cats[:, 0]), self.emb_dept(cats[:, 1]),
                         self.emb_store(cats[:, 2])], dim=-1)
        return self.static_grn(emb)  # [B, hidden]

    def forward(self, past_scaled, past_covs, fut_covs, cats, return_attn=False):
        static = self._static(cats)                      # [B, hid]
        enc_raw = torch.cat([past_scaled.unsqueeze(-1), past_covs], dim=-1)
        enc_in = self.enc_proj(enc_raw, static)          # [B, C, hid]
        dec_in = self.dec_proj(fut_covs, static)         # [B, H, hid]

        enc_out, (h, c) = self.lstm_enc(enc_in)
        dec_out, _ = self.lstm_dec(dec_in, (h, c))
        seq = self.gate_lstm(torch.cat([enc_out, dec_out], dim=1))  # [B, C+H, hid]

        C, H = self.context_len, dec_out.shape[1]
        q = seq[:, C:, :]                                # decoder positions as queries
        mask = self._attn_mask(C, H, seq.device)
        attn_out, attn_w = self.attn(q, seq, seq, attn_mask=mask,
                                     need_weights=return_attn, average_attn_weights=True)
        z = self.attn_grn(attn_out + dec_out)            # residual around attention
        z = self.pos_grn(z)
        preds = self.head(z)                             # [B, H, Q]  (scaled units)
        return (preds, attn_w) if return_attn else preds

    def quantile_loss(self, preds, target_scaled):
        """Pinball loss averaged over batch, horizon, quantiles. preds [B,H,Q]."""
        y = target_scaled.unsqueeze(-1)
        q = self.quantiles.view(1, 1, -1)
        e = y - preds
        return torch.maximum(q * e, (q - 1) * e).mean()
