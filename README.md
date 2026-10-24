# calibrate-bench

An independent, reproducible audit of whether decision models and LLMs are as calibrated as their vendors say. See [SPEC.md](SPEC.md).

**Status:** M0 (dataset loaders + license manifest) and the ECE / Brier / AURC part of M2, with soft-label support. No model runners yet.

```bash
pip install -e ".[dev]"
pytest -q                                  # offline tests
CBENCH_NETWORK=1 pytest -q                 # also downloads the datasets and checks split fingerprints
python -m cbench.datasets                  # print the license manifest and load everything
```

## Datasets

Each dataset is pinned to an immutable revision. Splits: dedupe on text, drop texts with conflicting labels, order by `sha256(dataset:id)`, then the first 500 rows become `calib` and the next 2000 `test`. Because the order comes from a hash, it doesn't depend on upstream row order. A fingerprint test catches any change.

| Slot | Source | License (verified 2026-10-23) | Notes |
|---|---|---|---|
| sentiment_hard | DynaSent v1.1, R1+R2 test | CC BY 4.0 | Adversarially collected, 5 validators per item |
| intent_routing | CLINC150 `plus`, test | CC BY 3.0 | 151 classes, ~17% out-of-scope |
| toxicity | Civil Comments, test | CC0 1.0 | label = rater fraction ≥ 0.5, ~8% positive. **Contamination risk:** Liquid AI lists Civil Comments among d1-3B's own evals |
| code_review_risk | CodeReviewer quality estimation, test | CC BY 4.0 (Zenodo 6900648) | Label is "a reviewer commented", i.e. observed behaviour, not curated annotation; 1.5 GB download |
| ticket_triage | Banking77, test (**stand-in**) | CC BY 4.0 (upstream LICENSE, verified 2026-10-24) | Single-sentence banking queries, not real tickets, and similar in kind to intent_routing. No human-labeled, permissively licensed ticket set exists that I could find; the most used HF option is synthetic and CC BY-NC |

`sentiment_hard` and `toxicity` also carry a `soft` column: the rater distribution over options. On DynaSent only 49% of items are unanimous (mean majority share 0.90); on Civil Comments 71%.

## Metrics (`cbench.metrics`)

- `ece(probs, labels, n_bins=15)`: equal-mass bins, per SPEC §3.
- `brier(probs, labels)`: multiclass, sum over classes, range [0, 2]. The binary convention (`(p - y)^2` on the positive class only) is exactly half of this.
- `risk_coverage`, `aurc`: tied confidences are accepted or rejected together, so results don't depend on input order.
- `bootstrap_ci(metric, probs, labels)`: seeded percentile bootstrap.

Every metric also accepts soft targets of shape (n, K). "Correct" then means rater agreement with the prediction, so ECE asks "is 70% confidence right where 70% of raters agree?". Soft-target scores have a non-zero floor (the raters' own disagreement), so they go in their own leaderboard column, never ranked against hard-target scores.

Inputs are validated (shape, label range, rows that are finite distributions), because they will come from submitted files.

## Verification log (SPEC §10, checked 2026-10-24)

| Item | Finding | Source |
|---|---|---|
| d1-3B schema | `model.system_one(state, questions)` with Decision Index questions (`type`: `noul` / `choice` / `score`, `instructions`, `criteria`). `choice` returns `choice`, `confidence`, `probabilities` over named options; `noul` returns only P(yes). Needs `transformers>=5.14` and `trust_remote_code=True`. LFM Open License v1.0: free below $10M annual revenue, no benchmarking restriction. Pin: `LiquidAI/d1-3B@051bcc46` | [model card](https://huggingface.co/LiquidAI/d1-3B), LICENSE in the repo |
| Decider weights | Public, Apache-2.0 (`Mapika/decider-2b@7614a6c7`, plus 4b/12b/31b/35b-a3b). **v12 (2026-10-20) replaced v11 in place**, and its default config sets yes/no temperature to 0.05, which its own card says makes yes/no probabilities uncalibrated (ECE 0.341 vs 0.095 at the fitted temperature). Its training mix includes train splits of Decision Index benchmarks. Runners must pin both the revision and the temperature config | [model card](https://huggingface.co/Mapika/decider-2b) |
| Winnow weights | Public, Apache-2.0: `EldanRing/Winnow-12B@0b5b7f98`, `Winnow-E4B@17871fd2` (Gemma 4 fine-tunes) | HF repo metadata |
| Jev terms | **Unresolved.** TypeSafe's public Terms (updated 2026-09-19) cover the website only and defer to a separate product agreement that isn't public; the Acceptable Use Policy (2026-09-23) bans reverse engineering but says nothing on benchmarking or publishing results. Get written confirmation from TypeSafe before publishing Jev numbers | typesafe.ai/legal/terms, typesafe.ai/legal/acceptable-use-policy |

Also found: `perplexity-ai/pplx-decider-*` (Apache-2.0), another open decision-model family worth adding in v1.
