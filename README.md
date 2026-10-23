# calibrate-bench

An independent, reproducible audit of whether decision models and LLMs are as calibrated as their vendors say. See [SPEC.md](SPEC.md).

**Status:** M0 (dataset loaders + license manifest) and the ECE / Brier / AURC part of M2. No model runners yet.

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
| toxicity | Civil Comments, test | CC0 1.0 | label = rater fraction ≥ 0.5, ~8% positive |
| code_review_risk | CodeReviewer quality estimation, test | CC BY 4.0 (Zenodo 6900648) | Label is "a reviewer commented", i.e. observed behaviour, not curated annotation; 1.5 GB download |
| ticket_triage | — | **blocked** | No human-labeled, permissively licensed set found. The most used HF option is synthetic and CC BY-NC |

So M0's "5 datasets load" is met for 4 of 5.

## Metrics (`cbench.metrics`)

- `ece(probs, labels, n_bins=15)`: equal-mass bins, per SPEC §3.
- `brier(probs, labels)`: multiclass, sum over classes, range [0, 2]. The binary convention (`(p - y)^2` on the positive class only) is exactly half of this.
- `risk_coverage`, `aurc`: tied confidences are accepted or rejected together, so results don't depend on input order.
- `bootstrap_ci(metric, probs, labels)`: seeded percentile bootstrap.

Inputs are validated (shape, label range, rows that are finite distributions), because they will come from submitted files.

## Still to verify (SPEC §10)

d1-3B output schema, whether Decider/Winnow weights are public, Jev terms on publishing benchmarks. None of these were checked yet.
