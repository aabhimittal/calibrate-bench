"""Dataset loaders and license manifest (M0).

Every source is pinned to an immutable revision (git commit or sha256), and its
license was checked by hand against the evidence in `license_evidence` on VERIFIED.
Splits are assigned by hashing example text, so they don't depend on upstream row order.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

CACHE = Path(os.environ.get("CBENCH_CACHE", Path.home() / ".cache" / "cbench"))
VERIFIED = "2026-10-23"
N_CALIB, N_TEST = 500, 2000


@dataclass(frozen=True)
class Source:
    name: str
    task: str
    license: str
    license_evidence: str
    origin: str
    revision: str
    citation: str
    notes: str = ""
    blocked: str = ""  # non-empty = reason this slot has no loader


@dataclass(frozen=True)
class Dataset:
    source: Source
    options: tuple[str, ...]
    df: pd.DataFrame  # columns: id, text, label (index into options), split ("calib" | "test")


SOURCES = {
    s.name: s
    for s in [
        Source(
            name="sentiment_hard",
            task="3-way sentiment on adversarially collected sentences",
            license="CC-BY-4.0",
            license_evidence="'License' section of the dataset README and data_license in "
            "hf.co/datasets/dynabench/dynasent/dynasent.py; upstream github.com/cgpotts/dynasent",
            origin="https://raw.githubusercontent.com/cgpotts/dynasent/"
            "707b6208978bfe7ff251a0c3263c9f0552c27b48/dynasent-v1.1.zip",
            revision="sha256:33001cf394618aa38f9530c43ca87072b92f5ee609a02afa2d168d25560cedfd",
            citation="Potts et al. 2021, DynaSent: A Dynamic Benchmark for Sentiment Analysis (ACL)",
            notes="Round 1 + Round 2 test files; 5 MTurk validators, gold = majority of >=3; "
            "'mixed' and no-majority examples dropped.",
        ),
        Source(
            name="intent_routing",
            task="151-way intent routing incl. out-of-scope",
            license="CC-BY-3.0",
            license_evidence="license tag on hf.co/datasets/clinc/clinc_oos",
            origin="hf://datasets/clinc/clinc_oos/plus/test-00000-of-00001.parquet",
            revision="155b9c710419136e17307b80d0a13e68cd46b4ec",
            citation="Larson et al. 2019, An Evaluation Dataset for Intent Classification and "
            "Out-of-Scope Prediction (EMNLP)",
            notes="'plus' config, test split. Queries crowdsourced, labels by construction of the prompt.",
        ),
        Source(
            name="toxicity",
            task="binary toxicity",
            license="CC0-1.0",
            license_evidence="license tag on hf.co/datasets/google/civil_comments",
            origin="hf://datasets/google/civil_comments/data/test-00000-of-00001.parquet",
            revision="f2970eb3a55777454c94069077cc8d9b5866312d",
            citation="Borkan et al. 2019, Nuanced Metrics for Measuring Unintended Bias (WWW)",
            notes="toxicity is the fraction of raters saying toxic; label = toxicity >= 0.5 "
            "(Jigsaw convention). Natural ~8% positive rate is kept.",
        ),
        Source(
            name="code_review_risk",
            task="binary: does this diff hunk draw a reviewer comment",
            license="CC-BY-4.0",
            license_evidence="zenodo.org/records/6900648 metadata (upstream release); "
            "the HF mirror used for download has no card of its own",
            origin="hf://datasets/fasterinnerlooper/codereviewer/quality/cls-test.jsonl",
            revision="555f3cda18bb62f90e322bb6b67d51b4a6814af5",
            citation="Li et al. 2022, Automating Code Review Activities by Large-Scale Pre-Training (FSE)",
            notes="Label is revealed human behaviour (a reviewer commented), not a curated annotation. "
            "Patches come from GitHub repos under their own licenses. Download is 1.5 GB.",
        ),
        Source(
            name="ticket_triage",
            task="support-ticket routing",
            license="",
            license_evidence="",
            origin="",
            revision="",
            citation="",
            blocked="No human-labeled, permissively licensed candidate found. The most used HF "
            "option (Tobi-Bueck/customer-support-tickets) is synthetic and CC-BY-NC-4.0.",
        ),
    ]
}


def _fetch(url: str, sha256: str) -> Path:
    path = CACHE / sha256
    if not path.exists():
        data = urllib.request.urlopen(url, timeout=300).read()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError(f"checksum mismatch for {url}")
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


def _hf(src: Source) -> str:
    owner, repo, filename = src.origin.removeprefix("hf://datasets/").split("/", 2)
    return hf_hub_download(f"{owner}/{repo}", filename, repo_type="dataset", revision=src.revision)


def _sentiment_hard(src: Source):
    options = ("negative", "neutral", "positive")
    z = zipfile.ZipFile(_fetch(src.origin, src.revision.removeprefix("sha256:")))
    rows = [
        json.loads(line)
        for r in ("round01-yelp", "round02-dynabench")
        for line in z.open(f"dynasent-v1.1/dynasent-v1.1-{r}-test.jsonl")
    ]
    df = pd.DataFrame({"text": [r["sentence"] for r in rows], "gold": [r["gold_label"] for r in rows]})
    df = df[df.gold.isin(options)]
    return options, df.assign(label=df.gold.map(options.index))


def _intent_routing(src: Source):
    path = _hf(src)
    options = tuple(json.loads(pq.read_schema(path).metadata[b"huggingface"])["info"]["features"]["intent"]["names"])
    return options, pd.read_parquet(path).rename(columns={"intent": "label"})


def _toxicity(src: Source):
    df = pd.read_parquet(_hf(src), columns=["text", "toxicity"])
    return ("non_toxic", "toxic"), df.assign(label=(df.toxicity >= 0.5).astype(int))


def _code_review_risk(src: Source):
    rows = []
    with open(_hf(src), encoding="utf-8") as f:
        for line in f:  # stream: each record also carries the full old file, which we don't keep
            r = json.loads(line)
            rows.append((r["patch"], int(r["y"])))
    return ("no_comment", "needs_comment"), pd.DataFrame(rows, columns=["text", "label"])


LOADERS = {
    "sentiment_hard": _sentiment_hard,
    "intent_routing": _intent_routing,
    "toxicity": _toxicity,
    "code_review_risk": _code_review_risk,
}


def assign_splits(df: pd.DataFrame, name: str, n_calib: int = N_CALIB, n_test: int = N_TEST) -> pd.DataFrame:
    """Dedupe on text, then take a hash-ordered sample: first n_calib rows -> calib, next n_test -> test."""
    n_labels = df.groupby("text")["label"].nunique()
    df = df[df.text.map(n_labels) == 1].drop_duplicates("text")  # texts with conflicting labels are dropped
    ids = df.text.map(lambda t: hashlib.sha256(t.encode()).hexdigest()[:16])
    key = ids.map(lambda i: hashlib.sha256(f"{name}:{i}".encode()).hexdigest())
    df = df.assign(id=ids, _key=key).sort_values("_key").head(n_calib + n_test)
    df["split"] = ["calib" if i < n_calib else "test" for i in range(len(df))]
    return df[["id", "text", "label", "split"]].reset_index(drop=True)


def load(name: str) -> Dataset:
    src = SOURCES[name]
    if src.blocked:
        raise RuntimeError(f"{name} is blocked: {src.blocked}")
    options, df = LOADERS[name](src)
    return Dataset(src, options, assign_splits(df[["text", "label"]], name))
