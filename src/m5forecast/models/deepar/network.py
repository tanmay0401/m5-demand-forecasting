"""DeepAR network: embeddings + LSTM + Negative Binomial head.

The architectural idea in one sentence: an LSTM reads (yesterday's scaled
sales, today's known covariates, series embeddings) and emits, per day,
the two parameters (mu, alpha) of a Negative Binomial distribution over
today's sales — the network predicts DISTRIBUTIONS, and training just
maximizes the likelihood of what actually happened.

Why Negative Binomial (paper's choice for retail counts): discrete,
non-negative, with a dispersion parameter alpha so variance = mu + alpha*mu^2
can exceed the mean — Poisson can't, and 73%-zeros-with-occasional-10x-spikes
data is wildly overdispersed.

Scale handling on the head (paper §3.3): mu is multiplied by the window's
scale nu, alpha divided by sqrt(nu) — the network learns scale-free shapes,
the head restores units.
"""

from __future__ import annotations

import torch
from torch import nn


class DeepARNet(nn.Module):
    def __init__(self, n_items: int, n_depts: int, n_stores: int, cov_dim: int,
                 hidden: int = 64, layers: int = 2, dropout: float = 0.1, embed_dim: int = 16):
        super().__init__()
        self.emb_item = nn.Embedding(n_items, embed_dim)
        self.emb_dept = nn.Embedding(n_depts, 4)
        self.emb_store = nn.Embedding(n_stores, 4)
        in_dim = 1 + cov_dim + embed_dim + 4 + 4  # scaled sales + covs + embeddings
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head_mu = nn.Linear(hidden, 1)
        self.head_alpha = nn.Linear(hidden, 1)

    def _step_inputs(self, prev_scaled, covs, cats):
        """[B, T] prev sales + [B, T, C] covs + static embeddings tiled over T."""
        b, t = prev_scaled.shape
        emb = torch.cat(
            [self.emb_item(cats[:, 0]), self.emb_dept(cats[:, 1]), self.emb_store(cats[:, 2])], dim=-1
        )
        emb = emb.unsqueeze(1).expand(b, t, emb.shape[-1])
        return torch.cat([prev_scaled.unsqueeze(-1), covs, emb], dim=-1)

    def _distribution(self, h, nu):
        """LSTM outputs -> NegBin(mu, alpha), scale-adjusted by nu [B]."""
        nu = nu.unsqueeze(-1)  # [B,1] broadcasts over time
        mu = torch.nn.functional.softplus(self.head_mu(h).squeeze(-1)) * nu + 1e-6
        alpha = torch.nn.functional.softplus(self.head_alpha(h).squeeze(-1)) / nu.sqrt() + 1e-6
        # torch parametrization: total_count r = 1/alpha; logits = log(mu/r) = log(mu*alpha)
        r = 1.0 / alpha
        logits = (mu * alpha).log()
        return torch.distributions.NegativeBinomial(total_count=r, logits=logits)

    def loss(self, batch) -> torch.Tensor:
        """Teacher-forced NLL over context + horizon (all steps supervise)."""
        # sequence of "previous day's scaled sales" inputs across context+horizon
        target_scaled = batch["target"] / batch["nu"].unsqueeze(-1)
        prev = torch.cat([batch["past_scaled"], target_scaled[:, :-1]], dim=1)
        covs = torch.cat([batch["past_covs"], batch["fut_covs"]], dim=1)[:, 1:, :]
        prev = prev[:, : covs.shape[1]]

        h, _ = self.lstm(self._step_inputs(prev, covs, batch["cats"]))
        dist = self._distribution(h, batch["nu"])

        # observed values for each step: past (from step 2) then the horizon
        obs = torch.cat([batch["past_scaled"][:, 1:] * batch["nu"].unsqueeze(-1), batch["target"]], dim=1)
        return -dist.log_prob(obs.round()).mean()

    @torch.no_grad()
    def sample_paths(self, past_scaled, past_covs, fut_covs, cats, nu, n_samples: int) -> torch.Tensor:
        """Ancestral sampling: encode context, then day-by-day sample->feed back.

        Returns [B, n_samples, horizon] integer-valued sample paths.
        """
        b, horizon = past_scaled.shape[0], fut_covs.shape[1]

        # encode context (inputs shifted: prev sales at t = sales[t-1])
        enc_in = self._step_inputs(past_scaled[:, :-1], past_covs[:, 1:, :], cats)
        _, state = self.lstm(enc_in)

        # tile everything across samples: [B*S, ...]
        s = n_samples
        h0 = state[0].repeat_interleave(s, dim=1).contiguous()
        c0 = state[1].repeat_interleave(s, dim=1).contiguous()
        state = (h0, c0)
        cats_t = cats.repeat_interleave(s, dim=0)
        nu_t = nu.repeat_interleave(s, dim=0)
        prev = past_scaled[:, -1].repeat_interleave(s, dim=0)  # last context day

        out = torch.zeros(b * s, horizon, device=past_scaled.device)
        for t in range(horizon):
            covs_t = fut_covs[:, t, :].repeat_interleave(s, dim=0).unsqueeze(1)
            x = self._step_inputs(prev.unsqueeze(1), covs_t, cats_t)
            h, state = self.lstm(x, state)
            dist = self._distribution(h.squeeze(1).unsqueeze(1), nu_t)
            draw = dist.sample().squeeze(1).clamp(max=5000)  # guard absurd tails
            out[:, t] = draw
            prev = draw / nu_t  # feed the SAMPLE back — uncertainty compounds honestly
        return out.view(b, s, horizon)
