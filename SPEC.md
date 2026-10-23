# CALIBRATE-BENCH — an independent audit of decision-model calibration

## 1. Thesis

Decision models sell one property above all: calibrated probabilities you can branch on. That claim is currently backed mostly by evaluations the vendors ran themselves. CALIBRATE-BENCH is an independent, reproducible audit of calibration, consistency and robustness across decision models and LLMs, with human-labeled ground truth and a PR-based leaderboard.

Analogy: a weights-and-measures inspector. Shops label their own scales; someone independent checks that 1 kg is 1 kg.

## 2. Why now (and why it's needed)

- TypeSafe's workflow evals use the average of two frontier models' predictions as the reference answer — they measure agreement with frontier LLMs, not ground truth. TypeSafe states this nuance openly on its blog.
- Liquid AI ran the Decision Index scorer itself for d1; third-party reports note these are not leaderboard submissions.
- Several decision models now exist (d1-3B, d1-omni-600M, Decider, Winnow, Jev), so a comparison is newly possible.

## 3. What gets measured

| Property | Metric | Why it matters |
|---|---|---|
| Calibration | ECE with equal-mass bins, Brier score, reliability diagrams | "70% confident" should be right 70% of the time |
| Selective prediction | Risk-coverage curve, AURC | Can you automate the confident fraction safely? |
| Consistency | Paraphrase KL: mean KL between answer distributions for semantically equivalent states | TypeSafe claims "similar answers for similar inputs" |
| Option-order sensitivity | Flip rate when choice options are permuted | Positional bias is common in LLMs |
| Robustness | Minimal-edit flip rate under adversarial perturbation | Guardrail use cases depend on this |
| Cost & latency | p50/p99, $ per 1k decisions | The whole value proposition |

ECE definition used: partition predictions into M equal-mass bins B_m; ECE = Σ_m (|B_m|/n)·|acc(B_m) − conf(B_m)|. Report with bootstrap CIs.

## 4. Scope

- MVP: 5 human-labeled datasets (sentiment-hard, intent routing, toxicity, ticket triage, code-review risk), 3 models (d1-3B, one open LLM via logprobs, one API LLM).
- v1: all available decision models, Jev if access granted, leaderboard site, PR submission flow.
- Non-goals: general capability benchmarking; generation quality.

## 5. Architecture

```
datasets/ (human-labeled, license-checked) ─▶ task adapters ─▶ typed questions
                                                         ▼
                                     model runners (decision models, LLM logprob runner)
                                                         ▼
                                     predictions.parquet (prob vectors, latency)
                                                         ▼
                    metrics (calibration, consistency, robustness) ─▶ leaderboard JSON ─▶ static site
```

Perturbation generators: paraphrase (LLM-generated, human-spot-checked), option permutation, adversarial minimal edits (character/word level).

## 6. Leaderboard mechanics (forks)

Submissions are PRs adding `submissions/<model>/predictions.parquet` + a manifest (model version, hardware, date). CI recomputes all metrics from raw predictions; nobody submits their own scores.

## 7. Milestones

- **M0** Dataset loaders with license manifest. Acceptance: 5 datasets load, splits are deterministic, licenses recorded.
- **M1** Runners for d1-3B and an LLM logprob runner. Acceptance: predictions file with full probability vectors.
- **M2** Calibration + selective metrics with bootstrap CIs. Acceptance: metrics match a hand-computed toy example exactly.
- **M3** Consistency + robustness perturbations.
- **M4** CI-verified submissions + static leaderboard.
- **M5** Write-up: what calibration claims hold, which don't.

## 8. Repo layout & stack

`cbench/ {datasets/, runners/, perturb/, metrics/, leaderboard/}`, Python, pandas/pyarrow, numpy, GitHub Actions, a static site (Astro or plain HTML).

## 9. Risks & honest weaknesses

- Human-labeled datasets may be in training data; report contamination checks where possible.
- Paraphrase generation can change meaning; spot-check a sample and report agreement.
- Vendors may update models silently; pin versions and dates.
- Jev access is gated; an audit without it is still useful but less newsworthy.

## 10. Verify before building

Dataset licenses; d1-3B schema; whether Decider/Winnow weights are actually public; Jev terms of use regarding publishing benchmarks.

## 11. Claude Code kickoff

"Read SPEC.md. Implement M0 and the ECE/Brier/AURC functions from M2 with tests against hand-computed toy cases. Verify dataset licenses before adding any loader."
