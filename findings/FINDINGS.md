# CALIBRATE-BENCH findings, round 1 (2026-10-26)

**Short version.**

- **d1-3B's calibration claim holds on hard sentiment, and its decision training is what earns it.** Run through the same prompt and readout, d1's own base model has ECE 0.281 and d1 has 0.077. Paired on the same 400 items, d1's Brier score is better by 0.214 (95% CI 0.145–0.278).
- **The training also removes option-order bias.** Shuffling the options flips 2.9% of d1's answers. It flips 15.9% for the same base model and 16.1% for a general 2B LLM.
- **On toxicity, calibration is not usefulness.** d1's ECE is a respectable 0.057, but it does not beat a predictor that always outputs the class frequencies (Brier difference +0.074, CI −0.012 to +0.161). It is also less accurate than always answering "non-toxic" (0.880 vs 0.953 on these items). Liquid's reported 93.0 on Civil Comments is 0.55 points above that always-"non-toxic" answer on the full test split.
- **It also holds on 77-way ticket routing:** ECE 0.055 at 0.890 accuracy, on 100 items.
- **Both general models are badly overconfident.** When the base model says 97% on sentiment it is right 57% of the time. Under our toxicity question, both flag most benign comments as toxic.

This is a first, small-sample audit run on a 4-core CPU with no GPU. Every number comes from fixed, shared subsets of the test splits, scored by the same code the leaderboard uses. The leaderboard marks these runs partial and unranked. Read the intervals, not the point estimates.

## What was tested

| # | Claim | Who makes it | Verdict here |
|---|---|---|---|
| 1 | d1-3B "returns calibrated, typed answers" | Liquid AI, d1-3B model card | **Supported on sentiment and 77-way ticket routing**, and on sentiment shown to be caused by the decision training. **Qualified on toxicity**: calibrated, but not better than knowing the base rate |
| 2 | d1-3B scores 93.0 on Civil Comments | d1-3B model card | **Uninformative**: 92.45% of the test split is non-toxic, so accuracy says little. d1 scored 0.880 on our 150 items |
| 3 | Answers stay put when options are reordered | Implied by "typed, calibrated decisions" | **Supported for d1** (2.9% flips), and caused by the training: the same base model through the same readout flips 15.9% |
| 4 | d1-3B leads the Decision Index | d1-3B model card ("not a leaderboard submission") | Not tested. The score is self-run; this is a provenance issue only |
| 5 | Decider gives calibrated probabilities | Decider model card | Not run. Its own card reports yes/no ECE 0.341 at the default temperature 0.05, versus 0.095 at the fitted one |
| 6 | Jev's evals | TypeSafe blog | Not run. The reference answer is two frontier models' average, and the API terms on publishing benchmarks are unresolved |

## Models

| Name | What it is | Readout |
|---|---|---|
| d1-3B | `LiquidAI/d1-3B@051bcc46`, the decision model under audit | Liquid's System One readout: one `choice` question, a one-token code per option, `calibration=None` (the vendor default) |
| LFM2.5-VL-3B via d1 | d1's base model, `LiquidAI/LFM2.5-VL-3B@35a118d9`, loaded with d1's config, code and tokenizer | Identical to d1. The only difference is Liquid's decision training. Loading d1's own weights the same way reproduces d1 to 6.5×10⁻⁷, and any missing or unexpected weight is an error |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B@15852e8c`, a general LLM of similar size and the base of Decider-2b | P(option text + end of turn), renormalised over the options |

All runs: fp32 on CPU, the dataset's fixed question, and options shown as words (`non toxic`, not `non_toxic`).

## Samples

The first N items of each test split in the benchmark's hash order, the same items for every model:
- 400 of `sentiment_hard` (DynaSent, adversarial 3-way sentiment);
- 150 of `toxicity` (Civil Comments);
- 100 of `ticket_triage` (Banking77, 77 options), d1 only. A matching `intent_routing` run (CLINC150, 151 options) was cut short by a machine restart and is not reported.

Every comparison between models is paired on these items, with 95% bootstrap intervals.

## The base-rate row comes first

A predictor that always outputs the class frequencies is perfectly calibrated and useless. Every table therefore carries a *base rate* row: the full test split's class frequencies, which a deployer would know without any model.

On Civil Comments, 92.45% of the test split is non-toxic, and 12% of comments split their raters 30–70%. Accuracy is the wrong yardstick for an 8%-positive task, so this audit leads with Brier scores, paired against that baseline.

## Results

### Calibration

Here ECE is expected calibration error and Brier is the Brier score, both 0 at best (Brier at most 2). "Top 50%" is accuracy on the half of answers given with the most confidence. Soft scores compare against the share of raters picking each option.

| dataset | model | n | accuracy | ECE [95% CI] | Brier [95% CI] | soft ECE | soft Brier | acc @ top 50% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sentiment_hard | **d1-3B** | 400 | 0.693 | **0.077** [0.064, 0.130] | **0.416** [0.365, 0.467] | 0.084 | 0.294 | 0.870 |
| sentiment_hard | LFM2.5-VL-3B via d1 | 400 | 0.613 | 0.281 [0.236, 0.327] | 0.630 [0.547, 0.707] | 0.300 | 0.498 | 0.760 |
| sentiment_hard | Qwen3.5-2B | 400 | 0.657 | 0.189 [0.154, 0.232] | 0.523 [0.459, 0.589] | 0.221 | 0.414 | 0.790 |
| sentiment_hard | *base rate* | 400 | 0.302 | 0.034 | 0.667 | | | |
| toxicity | **d1-3B** | 150 | 0.880 | **0.057** [0.045, 0.116] | **0.164** [0.109, 0.228] | 0.043 | 0.107 | 1.000 |
| toxicity | LFM2.5-VL-3B via d1 | 150 | 0.547 | 0.247 [0.200, 0.338] | 0.603 [0.500, 0.709] | 0.231 | 0.492 | 0.653 |
| toxicity | Qwen3.5-2B | 150 | 0.093 | 0.805 [0.757, 0.855] | 1.529 [1.436, 1.613] | 0.780 | 1.397 | 0.080 |
| toxicity | *base rate* | 150 | 0.953 | 0.029 | 0.091 | | | |

**Many-option routing (d1 only).** On 100 Banking77 queries with 77 options, d1-3B is accurate and well calibrated:

| dataset | model | n | accuracy | ECE [95% CI] | Brier [95% CI] | acc @ top 50% |
|---|---|---:|---:|---:|---:|---:|
| ticket_triage | **d1-3B** | 100 | 0.890 | **0.055** [0.054, 0.146] | 0.194 [0.110, 0.295] | 1.000 |
| ticket_triage | *base rate* | 100 | 0.040 | 0.023 | 0.986 | |

Its reliability bins are 0.62→0.70, 0.85→0.80, 0.95→0.95, 0.97→1.00, 0.98→1.00, and the paired Brier gain over the base rate is 0.793 [0.697, 0.877]. There is no general-model comparison here: the logprob runner scores every option separately, which takes minutes per item on CPU at 77 options.

### Paired Brier differences (same items; negative means the first is better)

| dataset | comparison | difference [95% CI] | reading |
|---|---|---:|---|
| sentiment_hard | d1-3B − its base model | **−0.214** [−0.278, −0.145] | Decision training improves the probabilities |
| sentiment_hard | d1-3B − Qwen3.5-2B | −0.107 [−0.166, −0.050] | d1 beats a general LLM |
| sentiment_hard | d1-3B − base rate | −0.252 [−0.302, −0.203] | d1 adds real information |
| sentiment_hard | base model − base rate | −0.038 [−0.116, +0.039] | Without the training, no clear gain over the prior |
| toxicity | d1-3B − its base model | **−0.439** [−0.524, −0.354] | Decision training improves the probabilities |
| toxicity | d1-3B − base rate | +0.074 [−0.012, +0.161] | **d1 does not beat knowing the prior** |
| toxicity | base model − base rate | +0.513 [+0.392, +0.631] | Worse than the prior |
| toxicity | Qwen3.5-2B − base rate | +1.438 [+1.298, +1.563] | Far worse than the prior |

### Reliability (five equal-mass bins: mean confidence → accuracy)

| dataset | model | bins |
|---|---|---|
| sentiment_hard | d1-3B | 0.50→0.45, 0.64→0.53, 0.76→0.75, 0.86→0.86, 0.94→0.88 |
| sentiment_hard | LFM2.5-VL-3B via d1 | 0.62→0.41, 0.88→0.47, 0.97→0.57, 1.00→0.72, 1.00→0.88 |
| sentiment_hard | Qwen3.5-2B | 0.55→0.41, 0.76→0.59, 0.94→0.60, 0.99→0.85, 1.00→0.84 |
| toxicity | d1-3B | 0.71→0.60, 0.88→0.80, 0.95→1.00, 0.97→1.00, 0.98→1.00 |
| toxicity | LFM2.5-VL-3B via d1 | 0.57→0.43, 0.69→0.43, 0.82→0.53, 0.91→0.63, 0.97→0.70 |
| toxicity | Qwen3.5-2B | 0.69→0.20, 0.89→0.00, 0.95→0.07, 0.98→0.07, 0.99→0.13 |

d1 is close to the diagonal on sentiment. On toxicity it is slightly overconfident in its least confident fifth and *under*confident at the top: its 95–98% answers were all correct. Both general models are overconfident everywhere.

### Option order (400 sentiment items × 2 reorderings)

| model | answers flipped | KL | items flipped by some reordering |
|---|---:|---:|---:|
| **d1-3B** | **2.9%** | **0.003** | 4.0% |
| LFM2.5-VL-3B via d1 | 15.9% | 0.271 | 19.0% |
| Qwen3.5-2B | 16.1% | 0.199 | 19.0% |

d1 scores single-letter option codes, the readout most exposed to position bias. Its base model, scored the same way, flips about one answer in six, so the order robustness comes from Liquid's training.

### Contamination probe (`cbench.contamination`, 50 test texts of 24+ tokens per dataset)

Each text is cut in half by tokens, and the probe counts how many of the following tokens the model's greedy continuation reproduces exactly. A run of 8 or more is flagged. The control is the MIT license, which code-trained models have seen thousands of times.

| model | MIT control | sentiment | toxicity | intent (12–13 long enough) | ticket | code review |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B | **42 / 42** | 0 flagged, max 2 | 0, max 3 | 0, max 3 | 0, max 3 | 5, max 43 |
| LFM2.5-VL-3B | **0 / 42** | 0, max 1 | 0, max 2 | 0, max 1 | 0, max 2 | 3, max 21 |
| d1-3B | **0 / 42** | 0, max 2 | 0, max 2 | 0, max 3 | 0, max 2 | 4, max 14 |

- **Qwen3.5-2B:** the probe works on this model, which recites the whole MIT continuation. It reproduces at most 3 tokens of any sentiment, toxicity, intent or ticket text, so there is **no sign of verbatim memorisation of those test sets**. That is still weak evidence: a model can learn a dataset's labels without memorising its wording.
- **LFM2.5-VL-3B and d1-3B:** the probe has **no power** on this model family. It doesn't continue the MIT license either, whether or not it gets a BOS token, and all weights load. So their clean rows say nothing either way. This matters most for toxicity, because Liquid lists Civil Comments among d1's own evaluations.
- **Code review flags are mostly predictable, not memorised:**
  - license headers (BOINC GPL, Apache);
  - import blocks;
  - one exponential-backoff idiom (`if m.currentStep > maxStep { m.currentStep = maxStep }` followed by a mirrored `Success()`), which a model can infer from the code above it.

  A flag on code needs a person to read it. This dataset's diffs come from public GitHub, so some pretraining overlap is expected anyway.
- Banking77 and CLINC queries are mostly shorter than 24 tokens. Verbatim continuation can't test them; only 12–13 CLINC items qualified.

## Things this audit got wrong on the way, and fixed

Each of these would have changed a headline number:

1. **ECE split tied confidences across bins** by input order. A constant base-rate predictor scored ECE 0.50 instead of 0, so the metric was measuring sort order. Ties now share a bin.
2. **The base-rate row used the scored sample's own class frequencies**, which leaks its labels. It now uses the full split.
3. **The weight override first compared checkpoint names with transformers' renamed ones**, and failed. Checkpoints now load through transformers itself, and missing weights are an error rather than silently random. An identity check confirms the loading.
4. **Option labels:** the LLM runner scored raw ids (`non_toxic`) while d1 saw words. Switching to words moved Qwen's toxicity Brier by only 0.010 (CI −0.005 to +0.022). The fix was for fairness; it did not explain Qwen's over-flagging.
5. **An out-of-memory kill was hidden** by a log filter. Runs now record exit codes.

## Limitations

- **Small samples on CPU.** There are 100–400 items per dataset, the intervals are wide, and the leaderboard ranks none of these runs. Full test splits need a GPU: d1-3B took about 5 s per short item, and over 25 s per item with 77+ options, on 4 CPU cores.
- **One question wording per dataset.** Both general models' toxicity over-flagging may depend on how the question is put. Wording was not varied.
- **Two readouts.** The base-model control removes this confound for the decision-training findings; the d1-vs-Qwen comparison keeps it.
- **One general LLM, at 2B parameters.** Larger general models may calibrate better.
- **Contamination.** Liquid lists Civil Comments among d1's own evaluations, and DynaSent, CLINC and Banking77 have been public for years. The verbatim probe clears Qwen on the text datasets but cannot test d1 or its base model (see above), and no probe here tests label memorisation.
- **Not covered this round:** paraphrase or typo runs, hosted models (no credentials), Decider, Winnow and Jev.

## Reproduce

```bash
pip install -e ".[runners]"
D="LiquidAI/d1-3B 051bcc464b01b9f92942b364d9586b0ef5912432"
python -m cbench.runners d1 $D sentiment_hard --limit 400
python -m cbench.runners d1 $D toxicity --limit 150 --batch-size 1      # batch 8 runs out of 15 GB RAM
python -m cbench.runners d1 $D sentiment_hard --limit 400 --perturb order --variants 2
python -m cbench.runners d1 $D sentiment_hard --limit 400 --out submissions/LFM2.5-VL-3B-via-d1 \
    --weights LiquidAI/LFM2.5-VL-3B@35a118d938ce6d123ac2d371649f24a8efb69058
python -m cbench.runners logprob Qwen/Qwen3.5-2B 15852e8c16360a2fea060d615a32b45270f8a8fc sentiment_hard --limit 400
python findings/analyze.py submissions/d1-3B submissions/LFM2.5-VL-3B-via-d1 submissions/Qwen3.5-2B
C="sentiment_hard toxicity intent_routing ticket_triage code_review_risk"
python -m cbench.contamination Qwen/Qwen3.5-2B 15852e8c16360a2fea060d615a32b45270f8a8fc $C
python -m cbench.contamination LiquidAI/LFM2.5-VL-3B 35a118d938ce6d123ac2d371649f24a8efb69058 $C
python -m cbench.contamination LiquidAI/d1-3B 051bcc464b01b9f92942b364d9586b0ef5912432 $C --remote-code
```

The base-model toxicity and option-order runs follow the same pattern. The prediction files behind these tables are submitted to the leaderboard separately, one pull request per model, so `analyze.py` reproduces every number above from `submissions/`.
