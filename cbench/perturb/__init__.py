"""Perturbations (M3): option order, character-level typos, paraphrases.

Everything is seeded from (dataset, item id), so every model sees the same perturbed inputs.
A variant's probabilities are always stored in the dataset's canonical option order.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd

PARAPHRASES = Path(__file__).parent / "paraphrases"
# Editing or paraphrasing a code diff changes what the reviewer would see, so the label may no longer hold.
NO_TEXT_PERTURBATION = {"code_review_risk"}


def _rng(*key) -> random.Random:
    return random.Random(hashlib.sha256(":".join(map(str, key)).encode()).hexdigest())


def orders(k: int, n: int, dataset: str) -> list[list[int]]:
    """n option orders that differ from the canonical one: reversed first, then seeded shuffles."""
    n = min(n, math.factorial(k) - 1)
    out, r = [list(range(k))[::-1]], _rng("order", dataset)
    while len(out) < n:
        p = list(range(k))
        r.shuffle(p)
        if p != sorted(p) and p not in out:
            out.append(p)
    return out[:n]


def typo(text: str, r: random.Random) -> str:
    """One keyboard-style typo (swap, drop, double or replace a letter) inside a word of 4+ letters.

    The first letter is kept, so the word stays recognisable. Returns text unchanged if no word qualifies.
    """
    words = list(re.finditer(r"[A-Za-z]{4,}", text))
    if not words:
        return text
    m = r.choice(words)
    w, i = m.group(), r.randrange(1, len(m.group()) - 1)
    op = r.choice(["swap", "drop", "double", "replace"])
    if op == "swap":
        w = w[:i] + w[i + 1] + w[i] + w[i + 2 :]
    elif op == "drop":
        w = w[:i] + w[i + 1 :]
    elif op == "double":
        w = w[:i] + w[i] + w[i:]
    else:
        w = w[:i] + r.choice([c for c in string.ascii_lowercase if c != w[i].lower()]) + w[i + 1 :]
    return text[: m.start()] + w + text[m.end() :]


def edits(text: str, n: int, dataset: str, item_id: str) -> list[str]:
    """n candidate minimal edits (one typo each) for an item: the attack budget."""
    return [typo(text, _rng("edit", dataset, item_id, k)) for k in range(n)]


def paraphrases(dataset: str) -> pd.DataFrame:
    """Stored paraphrases: columns id, k, text (empty if none were generated for this dataset)."""
    path = PARAPHRASES / f"{dataset}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["id", "k", "text"])


PARAPHRASE_PROMPT = (
    "Rewrite the text below in {k} different ways. Keep its meaning, sentiment, intent and tone exactly; "
    "change the wording and sentence structure, not the content. Do not fix or add anything.\n\nText:\n{text}"
)


def generate_paraphrases(dataset: str, n_items: int = 200, k: int = 3, model: str = "claude-opus-5-5",
                         client=None) -> Path:
    """Write k paraphrases for the first n_items test items to paraphrases/<dataset>.parquet.

    Generated once and committed, so every model is scored on the same paraphrases. Not usable as results
    until a human spot-check (spotcheck_sheet -> agreement) says they keep the meaning.
    """
    from cbench.datasets import load

    if dataset in NO_TEXT_PERTURBATION:
        raise ValueError(f"{dataset} is not paraphrased")
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    df = load(dataset).df
    df = df[df.split == "test"].head(n_items)
    keys = [f"p{j}" for j in range(k)]
    schema = {"type": "object", "properties": {key: {"type": "string"} for key in keys}, "required": keys,
              "additionalProperties": False}
    rows = []
    for item_id, text in zip(df.id, df.text):
        r = client.messages.create(
            model=model,
            max_tokens=4096,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(k=k, text=text)}],
        )
        if r.stop_reason != "end_turn":  # refused or cut off: no paraphrases for this item
            continue
        out = json.loads(next(b.text for b in r.content if b.type == "text"))
        rows += [(item_id, j, out[key], r.model) for j, key in enumerate(keys)]
    path = PARAPHRASES / f"{dataset}.parquet"
    path.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["id", "k", "text", "generator"]).to_parquet(path, index=False)
    return path


def spotcheck_sheet(dataset: str, n: int = 50) -> pd.DataFrame:
    """A seeded sample of (original, paraphrase) pairs with an empty same_meaning column for a human to fill."""
    from cbench.datasets import load

    para = paraphrases(dataset)
    orig = load(dataset).df.set_index("id").text
    sample = para.sample(n=min(n, len(para)), random_state=0)
    return sample.assign(original=sample.id.map(orig), same_meaning="")[["id", "k", "original", "text", "same_meaning"]]


def agreement(sheet: pd.DataFrame) -> tuple[float, int]:
    """Share of checked pairs a human marked as meaning-preserving (y/yes/1/true), and how many were checked."""
    marked = sheet.same_meaning.astype(str).str.strip().str.lower()
    checked = marked[marked != ""]
    return float(checked.isin(["y", "yes", "1", "true"]).mean()) if len(checked) else float("nan"), len(checked)


def report(preds: pd.DataFrame) -> pd.DataFrame:
    """Consistency and robustness per dataset and perturbation family, against each item's 'orig' row.

    flip_rate: share of (item, variant) pairs whose argmax differs from the original's.
    kl: mean KL(original || variant).
    any_flip: share of items where at least one variant flips the answer (attack success at this budget).
    """
    from cbench.metrics import kl, flips

    orig = preds[preds.variant == "orig"].set_index(["dataset", "id"]).probs
    rows = []
    var = preds[preds.variant != "orig"].assign(family=lambda d: d.variant.str.split(":").str[0])
    for (dataset, family), g in var.groupby(["dataset", "family"]):
        g = g[g.set_index(["dataset", "id"]).index.isin(orig.index)]
        if g.empty:
            continue
        p = np.stack(orig.loc[list(zip(g.dataset, g.id))].to_numpy())
        q = np.stack(g.probs.to_numpy())
        f = flips(p, q)
        rows.append({
            "dataset": dataset,
            "family": family,
            "items": g.id.nunique(),
            "variants": len(g),
            "flip_rate": float(f.mean()),
            "kl": float(kl(p, q).mean()),
            "any_flip": float(pd.Series(f).groupby(g.id.to_numpy()).max().mean()),
        })
    return pd.DataFrame(rows, columns=["dataset", "family", "items", "variants", "flip_rate", "kl", "any_flip"])
