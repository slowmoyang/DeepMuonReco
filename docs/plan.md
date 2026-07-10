# Plan: Improving `LatentCrossAttentionModel` Performance

Target model: `src/muonly/nn/models/latent_cross_attention.py`
Metric to maximize: **TNR at TPR ≥ 99.9%** (`BinarySpecificityAtSensitivity`, see `docs/metric.md`).

## Context and current design

Data flow in the current model:

1. Every object type (tracker track, DT/CSC/GEM segment, RPC/GEM hit) is projected
   to `model_dim` by a **single `nn.Linear`**.
2. Muon-detector measurements (~88 per event) are concatenated into one bag and
   compressed into `muon_det_latent_len` latents by a Perceiver encoder
   (`muon_det_encoder`, 1 cross-attention) + processor stack (`muon_det_processor`).
3. Each tracker track (~7000 per event) is scored by cross-attending its embedding
   to the latent muon memory (`decoder`), with **no track self-attention** (keeps
   cost linear in track count).
4. A `LayerNorm + Linear` head produces one logit per track.

Current config (`config/model/latent_cross_attention.yaml`):
`model_dim=128`, `num_heads=8`, `muon_det_latent_len=64`,
`muon_det_num_processors=1`, `decoder_num_layers=1`, `widening_factor=4`,
`dropout=0`. Optim: AdamW, `lr=3e-4`, `wd=1e-2`, `max_epochs=20`, cosine + warmup.
Loss: `BCEWithLogitsLoss` with `pos_weight=auto`.

### Key observations that shape the priorities

- **The metric only cares about the extreme tail.** TNR@TPR≥99.9% is set by the
  ~0.1-percentile *hardest positives* (the tracker muons the model scores lowest).
  Ordinary mean BCE spends almost all its gradient on easy examples and barely
  optimizes this operating point. **Loss design is likely the single biggest lever.**
- **The task is fundamentally geometric** (does the track extrapolate to a muon
  segment?), yet the model gets only raw preprocessed `px,py,pz,vx,vy,vz,charge`
  through one linear layer — no `eta/phi/pt`, no Fourier features, no per-detector
  identity. Feature/embedding richness is the second biggest lever.
- **The network is very shallow** (1 processor + 1 decoder layer, dim 128,
  dropout 0). There is headroom to scale depth/width — inference cost matters for
  production, but tune quality first, then trade back down.

---

## Priority 1 — Align the loss with the operating point (highest expected impact)

Mean BCE is a poor surrogate for TNR@TPR≥99.9%. Options, cheapest first:

1. **Hard-positive emphasis (focal / asymmetric loss).** Focal loss with
   `gamma≈1–2` and a positive-side focusing term concentrates gradient on the
   hardest positives that actually set the 99.9% threshold. Small, drop-in change
   to `compute_loss`/criterion in `scripts/train.py`.
2. **Partial-AUC (pAUC) surrogate in the high-TPR region.** Directly optimize the
   ranking only between positives and negatives near the top of the score
   distribution (e.g. maximize separation for FPR in the region that matters at
   TPR≈0.999). A soft/differentiable pAUC or a top-k pairwise ranking loss on
   in-batch positives-vs-hardest-negatives is well matched to the metric.
3. **Recall-constrained / sensitivity-at-fixed-threshold loss.** Penalize
   false negatives much more heavily than false positives (raise the effective
   `pos_weight`, or use an asymmetric margin), since losing a positive is what
   caps TPR. Sweep `pos_weight` above the `auto` value.
4. **Margin on the worst positives.** Add a term that pushes up the minimum (or
   soft-min / lowest-k) positive score per batch, directly attacking the tail that
   defines the threshold.

Recommendation: start with (1) focal + higher `pos_weight`, then try (2) pAUC.
Validate each against the true `sas_metric`, not training loss.

---

## Priority 2 — Richer inputs and per-object embeddings

Cheap, architecture-local changes with strong physics priors:

1. **Add derived kinematic features to tracker tracks.** Feed `pt, 1/pt, eta, phi`
   (and `cos phi, sin phi`) alongside the raw momenta. `eta/phi` define where a
   track extrapolates in the muon system — exactly the quantity cross-attention
   must match. Add in the dataset feature stack or compute inside the embedder.
2. **Fourier / positional feature encoding of geometry.** Replace the raw scaled
   position/momentum with random-Fourier or sinusoidal features before the linear
   projection. Coordinate-MLPs learn geometric matching far better from Fourier
   features than from raw coordinates.
3. **Per-detector type embedding.** All muon measurements are concatenated into one
   undifferentiated bag (`muon_det.py` lines 167–187). Add a learned type embedding
   (DT / CSC / GEM-seg / RPC-hit / GEM-hit) to each token so the latent encoder can
   tell subsystems apart. Also consider a learned embedding distinguishing
   "segment" (has direction) vs "hit" (position only).
4. **Deeper per-object embedder.** Replace the single `nn.Linear` embedders with a
   2-layer MLP (LayerNorm + GELU), giving the projection some nonlinearity before
   attention.
5. **Richer classification head.** The head is `LayerNorm + Linear`. Use a small
   MLP head (`LayerNorm → Linear → GELU → Linear`) for a more expressive decision
   boundary at negligible cost per track.

---

## Priority 3 — Transformer architecture optimization

The user-suggested direction. Ordered by expected value / cost:

1. **Increase decoder depth.** `decoder_num_layers=1` gives each track a single
   cross-attention read of the latent memory. Try 2–4 layers so tracks can refine
   their match iteratively. This is the most direct capacity increase on the
   per-track path.
2. **Increase processor depth.** `muon_det_num_processors=1` — a single
   self-attention pass over the latents. Try 2–4 processor layers so the muon-system
   summary is better organized before tracks read it.
3. **Tune latent count.** `muon_det_latent_len=64` compresses ~88 measurements. Try
   32 / 64 / 96 / 128; too few loses resolution, too many wastes compute. Sweep.
4. **Scale `model_dim` / heads.** `model_dim=128` (16 dims/head). Try 192/256; keep
   head-dim ≥ 32. Re-tune `num_heads` accordingly.
5. **Latent positional embedding.** Add learned positional embeddings to the latent
   array so processor self-attention can specialize latents to roles/regions.
6. **Gated / more expressive FFN.** Swap the MLP block for a SwiGLU-style gated FFN;
   often a free quality bump at equal parameter budget. Consider RMSNorm in place of
   LayerNorm for stability at depth.
7. **Regularization once deeper.** `dropout=0` is fine while shallow; when depth
   grows, sweep `dropout∈{0.05,0.1}` and attention dropout to control overfitting.
8. **(Cost-permitting) local track self-attention.** Full track self-attention is
   O(N²)≈7000² — too expensive. But *windowed / bucketed* self-attention among
   nearby tracks (by eta/phi) could let overlapping tracks share context. Only if
   the linear-cost budget allows; otherwise keep the no-self-attention design.

---

## Priority 4 — Training recipe

1. **Train longer.** `max_epochs=20` is short for a small model; extend to 40–60 and
   let cosine annealing finish, watching the val `sas_metric` (not just loss).
2. **Weight EMA.** Maintain an exponential moving average of weights for evaluation;
   typically improves tail metrics and stability at ~zero cost.
3. **LR / batch sweep.** Re-tune `lr` after any width/depth change; larger models
   often want lower LR. Sweep `weight_decay`.
4. **Threshold selection discipline.** The metric returns a candidate threshold on
   the validation set; freeze it and report on an independent test set
   (`docs/metric.md`). Ensure enough positive statistics to estimate the 0.1%
   inefficiency reliably — a noisy tail estimate makes model comparison unreliable.
5. **Checkpoint on `sas_metric`.** Select the best checkpoint by validation
   TNR@TPR≥99.9%, not by loss or AUROC.

---

## Suggested experiment order

1. **Baseline lock-in.** Record current val `sas_metric` + threshold as the
   reference. Ensure the metric is computed globally over all good tracks.
2. **Loss (P1):** focal + `pos_weight` sweep → then pAUC surrogate. *Biggest bet.*
3. **Features (P2):** add `pt,eta,phi` + detector-type embedding + MLP embedder/head.
4. **Depth (P3):** `decoder_num_layers` and `muon_det_num_processors` 1→2→4;
   then latent-count and `model_dim` sweeps.
5. **Recipe (P4):** longer training + EMA + LR re-tune on the best config.
6. **Cost trade-back.** Once quality is established, shrink to the smallest config
   within a target TNR, since production selection weighs both physics and timing
   (`docs/project.md`).

Change one axis at a time and compare on the true `sas_metric`. AUROC / loss are
only diagnostics — a model can improve AUROC while regressing the 99.9%-TPR tail.
