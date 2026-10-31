"""Contamination probe (SPEC §9): does a model continue test texts verbatim?

Each text is cut in half by tokens. One teacher-forced forward pass gives, at every position of the second half, the
model's top-1 next token; the length of the leading run where top-1 equals the real next token is exactly what greedy
decoding from the first half would reproduce word for word. Ordinary text diverges within a few tokens, so a run of
`k` or more tokens is hard to explain unless the text was in the training data.

The evidence is one-sided. A flagged item is strong evidence of memorisation (read the examples: short formulaic
texts like "I want to cancel my card" can be predictable without it). A clean result is weak evidence: a model can
learn a dataset's labels without memorising its wording, and there is no known-unseen control set here.

    python -m cbench.contamination MODEL SHA DATASET [DATASET ...] [--limit 50] [--remote-code]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cbench.datasets import load

# Positive control: the MIT license, copied into countless repos; a probe that can't flag this can't flag anything
CONTROL = ("Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated "
           "documentation files (the \"Software\"), to deal in the Software without restriction, including without "
           "limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the "
           "Software, and to permit persons to whom the Software is furnished to do so, subject to the following "
           "conditions:")


def verbatim_run(model, tok, text: str) -> tuple[int, int]:
    """(tokens greedy decoding reproduces after the first half, tokens in the second half)."""
    import torch

    ids = tok(text).input_ids
    cut = len(ids) // 2
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids], device=model.device)).logits[0]
    hit = (logits[cut - 1 : -1].argmax(-1).cpu() == torch.tensor(ids[cut:])).tolist()
    return (hit + [False]).index(False), len(hit)


def probe(model, tok, dataset: str, limit: int = 50, k: int = 8, min_tokens: int = 24) -> pd.DataFrame:
    """Verbatim runs on the first test texts with at least `min_tokens` tokens, so a k-token run is possible."""
    df = load(dataset).df
    df = df[df.split == "test"]
    df = df[[len(tok(t).input_ids) >= min_tokens for t in df.text]].head(limit)
    runs = [verbatim_run(model, tok, t) for t in df.text]
    return pd.DataFrame({"dataset": dataset, "id": df.id, "text": df.text, "run": [r for r, _ in runs],
                         "rest": [n for _, n in runs], "flagged": [r >= k for r, _ in runs]})


def summary(res: pd.DataFrame) -> pd.DataFrame:
    return res.groupby("dataset").agg(items=("run", "size"), flagged=("flagged", "sum"),
                                      median_run=("run", "median"), max_run=("run", "max"))


if __name__ == "__main__":
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    from cbench.runners import _snapshot

    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("revision")
    p.add_argument("datasets", nargs="+")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--remote-code", action="store_true", help="load with the repo's own AutoModel code, e.g. d1 "
                   "(pinned by the sha)")
    a = p.parse_args()
    path = _snapshot(a.model, a.revision)
    tok = AutoTokenizer.from_pretrained(path)
    # text-only class when transformers has one for this config (Qwen3.5), else the vision-language one (LFM2.5-VL)
    text_only = type(AutoConfig.from_pretrained(path)) in AutoModelForCausalLM._model_mapping
    cls = AutoModel if a.remote_code else AutoModelForCausalLM if text_only else AutoModelForImageTextToText
    model = cls.from_pretrained(path, dtype=torch.float32, trust_remote_code=a.remote_code).eval()
    res = pd.concat([probe(model, tok, ds, a.limit, a.k) for ds in a.datasets])
    print(f"## {a.model}@{a.revision[:12]}: greedy continuation of the second half, flagged at >= {a.k} tokens\n")
    run, rest = verbatim_run(model, tok, CONTROL)
    print(f"positive control (MIT license): run {run} of {rest} tokens")
    if run < a.k:
        print("WARNING: the control is not flagged, so this model doesn't reproduce even heavily copied text "
              "verbatim; a clean result below says nothing about contamination")
    print()
    print(summary(res).to_string())
    for _, r in res[res.flagged].iterrows():
        print(f"\n- {r.dataset} {r.id[:12]} ({r.run}/{r.rest} tokens): {r.text[:300]!r}")
    print(f"\nrun-length quantiles (50/90/99%): {np.quantile(res.run, [.5, .9, .99]).tolist()}")
