# calibrate-bench

An independent, reproducible audit of whether decision models and LLMs are as calibrated as their vendors say. See [SPEC.md](SPEC.md).

**Status:** M0 (dataset loaders + license manifest), M1 (d1, LLM logprob and Claude verbal runners), M3 (option-order, typo and paraphrase perturbations), M4 (CI-scored submissions and a static leaderboard), and the ECE / Brier / AURC part of M2, with soft-label support.

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

## Runners (`cbench.runners`)

```bash
pip install -e ".[runners]"
python -m cbench.runners d1 LiquidAI/d1-3B 051bcc464b01b9f92942b364d9586b0ef5912432 sentiment_hard toxicity
python -m cbench.runners logprob Qwen/Qwen3.5-2B 15852e8c16360a2fea060d615a32b45270f8a8fc sentiment_hard
pip install -e ".[api]"   # needs Anthropic credentials
python -m cbench.runners verbal claude-opus-5-5 sentiment_hard
```

Each run writes `submissions/<model>/predictions.parquet` (`dataset, id, variant, probs, latency_ms`, one full probability vector per item, in the dataset's option order; `variant` is `orig` or a perturbation, see below) and `manifest.json` (model, commit, readout, dtype, device, library versions, and per dataset: split, limit, batch size, date). Re-running a dataset replaces its rows; a directory never mixes two models or commits.

Every runner asks the dataset's fixed `question` from the manifest.

- **`d1`** uses Liquid's own System One readout: one `choice` question; each option gets a one-token code scored at a single answer slot; `calibration=None` (their default, so the probabilities are an uncalibrated softmax and any calibration comes from training).
- **`verbal`** is for hosted models with no token logprobs. The Claude Messages API has none (the live API reference never mentions them, and sampling parameters are gone too), so the model states a probability per option through a JSON schema, renormalised to 1. That is *verbalized* confidence, a different instrument from a logprob, so it is reported separately. Effort defaults to `low`; refusals are dropped and counted rather than handed to a fallback model, because a fallback would score another model under this one's name.
- **`logprob`** works with any causal LM: P(option text + end-of-turn token | chat prompt), renormalised over the options. The end-of-turn token keeps `no` from absorbing `no_comment`'s mass. It is read from the chat template (the token after an assistant reply), not from `generation_config`: Qwen3.5-0.8B ships no generation config, its config eos is `<|endoftext|>`, and scoring with that moved 9 of 20 toxicity answers in a smoke run. The manifest records the `end_token` used.

Revisions must be full commit SHAs. `trust_remote_code` runs the repo's Python, so a branch name would mean running whatever gets pushed next. Weights, tokenizer and code are loaded from one snapshot; d1's own loader fetches the tokenizer by repo name without a revision, and the snapshot closes that gap. I read d1's remote code at `051bcc46` before running it: it renders prompts and runs the model, with no network, subprocess or deserialization calls.

Latency is batch wall time divided by batch size; per-decision p50/p99 needs `--batch-size 1`. The logprob runner encodes the prompt once per item, batched 16 wide, and deep-copies that cache for each chunk of options. Hybrid models such as Qwen3.5 can't batch-repeat their linear-attention state, but every cache can be copied; scores match a full forward pass per option to 1e-5.

## Perturbations (`cbench.perturb`, M3)

```bash
python -m cbench.runners logprob MODEL SHA sentiment_hard --perturb order --variants 3
python -m cbench.runners logprob MODEL SHA sentiment_hard --perturb edit --variants 8
python -m cbench.perturb report submissions/MODEL/predictions.parquet
```

Each family is a separate run whose rows sit next to the `orig` rows; the report compares every variant with its item's original.

| Family | What changes | Measures |
|---|---|---|
| `order` | The options are shown in another order (reversed first, then seeded shuffles); probabilities are mapped back to the canonical order | Option-order sensitivity: flip rate and KL |
| `edit` | One keyboard typo (swap, drop, double or replace a letter, first letter kept) in a 4+ letter word; `--variants` is the attack budget | Robustness: per-edit flip rate, and `any_flip` = share of items where some edit in the budget flips the answer |
| `paraphrase` | Stored paraphrases from `python -m cbench.perturb paraphrase DATASET` (Claude, structured output) | Consistency: paraphrase KL and flip rate |

All perturbations are seeded by dataset and item, so every model sees identical inputs. `code_review_risk` gets order perturbations only: a typo or paraphrase in a diff changes the code, so the label may not hold.

Paraphrases can change meaning, so they are not results until a person has checked a sample: `python -m cbench.perturb spotcheck DATASET sheet.csv`, fill `same_meaning` with y/n, then `python -m cbench.perturb agreement sheet.csv` and report that number next to paraphrase scores. Paraphrases written by Claude may suit Claude's own style; score Claude's verbal runner on them with that caveat, or use another generator.

The edit family is a random-search attack with a fixed budget, not gradient or model-specific search, so `any_flip` is a lower bound on how attackable a model is. Word-level edits are not included: inserting or dropping words can change meaning ("not"), which a flip rate can't tell apart from fragility.

## Submitting to the leaderboard (`cbench.leaderboard`, M4)

1. Run a runner over the **full test split** of any datasets you choose (and optionally perturbations). It writes `submissions/<name>/predictions.parquet` and `manifest.json`.
2. Open a pull request that adds only those two files under one `submissions/<name>/` directory.
3. The `submission` workflow checks the PR touches nothing else, validates the files and posts the recomputed scores in the job summary. Nobody submits their own numbers: accuracy, ECE, Brier and AURC with 95% bootstrap intervals, soft-label scores, perturbation flip rates and latency all come from the raw probabilities.
4. After merge, the `leaderboard` workflow rebuilds the site with `main`'s scoring code and deploys it to GitHub Pages (enable once: Settings → Pages → Source: GitHub Actions).

```bash
python -m cbench.leaderboard check submissions/<name>      # what CI runs on a submission PR
python -m cbench.leaderboard build submissions site         # the site: site/index.html + site/leaderboard.json
```

Submissions that don't cover a dataset's whole test split are scored and shown, but marked partial and listed after the ranked ones. Complete submissions are ranked by Brier score, a proper scoring rule, within each dataset.

**How the scores stay honest.** A PR's CI normally runs the PR's own code, so a submission could also edit `cbench/metrics` and score itself perfectly. The submission workflow never runs PR code: it installs `cbench` from the base commit, and that copy checks the PR's changed paths (read NUL-separated, without rename detection) and scores its files as data. A PR that changes anything outside one submission directory, or adds any other file such as a scores file, is rejected. Labels come from `cbench/leaderboard/labels.parquet` (hashed ids, labels and rater distributions; no texts), so CI downloads no datasets; a network test regenerates it from the loaders and compares. Model names and other manifest text are HTML-escaped on the page.

## Verification log (SPEC §10, checked 2026-10-24)

| Item | Finding | Source |
|---|---|---|
| d1-3B schema | `model.system_one(state, questions)` with Decision Index questions (`type`: `noul` / `choice` / `score`, `instructions`, `criteria`). `choice` returns `choice`, `confidence`, `probabilities` over named options; `noul` returns only P(yes). Needs `transformers>=5.14` and `trust_remote_code=True`. LFM Open License v1.0: free below $10M annual revenue, no benchmarking restriction. Pin: `LiquidAI/d1-3B@051bcc46` | [model card](https://huggingface.co/LiquidAI/d1-3B), LICENSE in the repo |
| Decider weights | Public, Apache-2.0 (`Mapika/decider-2b@7614a6c7`, plus 4b/12b/31b/35b-a3b). **v12 (2026-10-20) replaced v11 in place**, and its default config sets yes/no temperature to 0.05, which its own card says makes yes/no probabilities uncalibrated (ECE 0.341 vs 0.095 at the fitted temperature). Its training mix includes train splits of Decision Index benchmarks. Runners must pin both the revision and the temperature config | [model card](https://huggingface.co/Mapika/decider-2b) |
| Winnow weights | Public, Apache-2.0: `EldanRing/Winnow-12B@0b5b7f98`, `Winnow-E4B@17871fd2` (Gemma 4 fine-tunes) | HF repo metadata |
| Jev terms | **Unresolved.** TypeSafe's public Terms (updated 2026-09-19) cover the website only and defer to a separate product agreement that isn't public; the Acceptable Use Policy (2026-09-23) bans reverse engineering but says nothing on benchmarking or publishing results. Get written confirmation from TypeSafe before publishing Jev numbers | typesafe.ai/legal/terms, typesafe.ai/legal/acceptable-use-policy |

Also found: `perplexity-ai/pplx-decider-*` (Apache-2.0), another open decision-model family worth adding in v1.
