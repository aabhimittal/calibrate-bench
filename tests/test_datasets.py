import hashlib
import os
import re

import numpy as np
import pandas as pd
import pytest

from cbench.datasets import LOADERS, N_CALIB, N_TEST, SOURCES, assign_splits, load


def _toy(n=50):
    return pd.DataFrame({"text": [f"example {i}" for i in range(n)], "label": [i % 3 for i in range(n)]})


def test_splits_ignore_upstream_row_order():
    df = _toy()
    a = assign_splits(df, "toy", n_calib=10, n_test=20)
    b = assign_splits(df.sample(frac=1, random_state=7), "toy", n_calib=10, n_test=20)
    pd.testing.assert_frame_equal(a, b)
    assert a.split.value_counts().to_dict() == {"calib": 10, "test": 20}


def test_splits_depend_on_dataset_name():
    df = _toy()
    assert not assign_splits(df, "a", 10, 20).id.equals(assign_splits(df, "b", 10, 20).id)


def test_small_dataset_fills_calib_first():
    assert assign_splits(_toy(5), "toy", n_calib=3, n_test=10).split.tolist() == ["calib"] * 3 + ["test"] * 2


def test_duplicates_deduped_and_conflicting_labels_dropped():
    df = pd.DataFrame({"text": ["a", "a", "b", "b", "c"], "label": [0, 0, 0, 1, 2]})
    assert sorted(assign_splits(df, "toy").text) == ["a", "c"]


def test_extra_columns_follow_their_rows():
    df = _toy(20).assign(soft=[[i, -i] for i in range(20)])
    out = assign_splits(df, "toy", n_calib=5, n_test=5)
    assert list(out.columns) == ["id", "text", "label", "soft", "split"]
    assert all(s[0] == int(t.split()[1]) for s, t in zip(out.soft, out.text))


def test_manifest_every_slot_is_pinned_and_licensed():
    assert set(SOURCES) == set(LOADERS) and len(SOURCES) == 5
    for name, src in SOURCES.items():
        assert src.license and src.license_evidence and src.citation, name
        assert re.fullmatch(r"[0-9a-f]{40}|sha256:[0-9a-f]{64}", src.revision), name


# Fingerprint of (id, label, split) for every loaded row. A change means the pinned data
# or the split procedure changed, which invalidates every submitted predictions file.
FINGERPRINTS = {
    "sentiment_hard": "68f3629c49e06527",
    "intent_routing": "01d404763583caf3",
    "toxicity": "603ff87fc9742859",
    "code_review_risk": "c83ebb8fdc2ee45a",
    "ticket_triage": "c4f3e4573478a386",
}
SOFT = {"sentiment_hard", "toxicity"}


@pytest.mark.skipif(not os.environ.get("CBENCH_NETWORK"), reason="set CBENCH_NETWORK=1 to download datasets")
@pytest.mark.parametrize("name", sorted(FINGERPRINTS))
def test_load_real_dataset(name):
    ds = load(name)
    df = ds.df
    soft = ["soft"] if name in SOFT else []
    assert list(df.columns) == ["id", "text", "label", *soft, "split"]
    assert df.split.value_counts().to_dict() == {"calib": N_CALIB, "test": N_TEST}
    assert df.id.is_unique and df.label.between(0, len(ds.options) - 1).all()
    if soft:
        targets = np.array(df.soft.tolist())
        assert targets.shape == (len(df), len(ds.options))
        np.testing.assert_allclose(targets.sum(1), 1)
        # the hard label must be (one of) the most-voted options
        assert (targets[np.arange(len(df)), df.label] >= targets.max(1) - 1e-9).all()
    key = ",".join(df.id + ":" + df.label.astype(str) + ":" + df.split)
    assert hashlib.sha256(key.encode()).hexdigest()[:16] == FINGERPRINTS[name]
