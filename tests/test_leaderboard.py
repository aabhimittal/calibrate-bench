import json
import os

import numpy as np
import pandas as pd
import pytest

from cbench import leaderboard as lb

LAB = lb.labels()


def make_sub(path, n=None, dataset="sentiment_hard", probs=None, manifest=None, extra=()):
    """A submission over the first n test ids of a dataset; probs default to 0.8 on the true label."""
    ids = LAB.loc[dataset].index[:n] if n else LAB.loc[dataset].index
    y = LAB.loc[dataset].loc[ids].label.to_numpy()
    k = 3 if dataset == "sentiment_hard" else 2
    if probs is None:
        probs = [[0.8 if j == t else 0.2 / (k - 1) for j in range(k)] for t in y]
    rows = [(dataset, i, "orig", p, 10.0) for i, p in zip(ids, probs)] + list(extra)
    path.mkdir(parents=True)
    pd.DataFrame(rows, columns=["dataset", "id", "variant", "probs", "latency_ms"]).to_parquet(
        path / "predictions.parquet")
    m = {"runner": "logprob", "model": "org/m", "revision": "a" * 40, "readout": "test",
         "runs": {dataset: {"batch_size": 1, "date": "2026-10-25"}}} | (manifest or {})
    (path / "manifest.json").write_text(json.dumps(m))
    return path, y


@pytest.mark.parametrize("changed", [
    ["submissions/m/predictions.parquet", "cbench/metrics/__init__.py"],  # also edits the scorer
    ["submissions/a/manifest.json", "submissions/b/manifest.json"],  # two submissions
    ["submissions/m/scores.json"],  # self-reported scores
    ["submissions/m/sub/predictions.parquet"],
    ["submissions/../cbench/manifest.json"],
    [],
])
def test_guard_rejects(changed):
    with pytest.raises(ValueError):
        lb.guard(changed)


def test_guard_accepts_one_submission():
    assert lb.guard(["submissions/d1-3B/predictions.parquet", "submissions/d1-3B/manifest.json"]) == "d1-3B"


def test_score_recomputes_from_probabilities(tmp_path):
    sub, y = make_sub(tmp_path / "m")
    s = lb.score(sub)["datasets"]["sentiment_hard"]
    assert s["n"] == 2000 and s["complete"] and s["accuracy"] == 1.0
    # every prediction puts 0.8 on the right answer: ECE = |1 - 0.8|, Brier = .2^2 + 2 * .1^2 = 0.06
    assert s["ece"][0] == pytest.approx(0.2) and s["brier"][0] == pytest.approx(0.06)
    assert s["ece"][1] <= s["ece"][0] <= s["ece"][2] and "soft_brier" in s


def test_partial_submission_is_scored_but_marked(tmp_path):
    sub, _ = make_sub(tmp_path / "m", n=50)
    assert lb.score(sub)["datasets"]["sentiment_hard"]["complete"] is False


@pytest.mark.parametrize("break_it, message", [
    (lambda d: (d / "manifest.json").unlink(), "missing manifest.json"),
    (lambda d: (d / "manifest.json").write_text(json.dumps({"model": "x"})), "no 'runner'"),
    (lambda d: pd.read_parquet(d / "predictions.parquet").assign(id="nope").drop_duplicates("id")
     .to_parquet(d / "predictions.parquet"), "not in the test split"),
    (lambda d: pd.read_parquet(d / "predictions.parquet").assign(variant="mine")
     .to_parquet(d / "predictions.parquet"), "unknown variants"),
    (lambda d: pd.read_parquet(d / "predictions.parquet").drop(columns="latency_ms")
     .to_parquet(d / "predictions.parquet"), "columns must be"),
    (lambda d: pd.concat([pd.read_parquet(d / "predictions.parquet")] * 2)
     .to_parquet(d / "predictions.parquet"), "duplicate"),
])
def test_validate_rejects(tmp_path, break_it, message):
    sub, _ = make_sub(tmp_path / "m", n=5)
    break_it(sub)
    with pytest.raises(ValueError, match=message):
        lb.validate(sub)


@pytest.mark.parametrize("field, value", [
    ("model", "m)\n\n| sentiment_hard | 2000 | True | 1.000 | 0.001 | 0.001 | 0.001 |"),  # forged summary row
    ("model", "a|b"),
    ("revision", "<img src=x>"),
    ("runner", "mine"),
])
def test_validate_rejects_unsafe_manifest_identifiers(tmp_path, field, value):
    sub, _ = make_sub(tmp_path / "m", n=5, manifest={field: value})
    with pytest.raises(ValueError, match=f"{field} must"):
        lb.validate(sub)


def test_validate_rejects_bad_probabilities_orphans_and_symlinks(tmp_path):
    sub, _ = make_sub(tmp_path / "a", n=5, probs=[[0.5, 0.5, 0.5]] * 5)
    with pytest.raises(ValueError, match="distribution"):
        lb.validate(sub)
    orphan = ("sentiment_hard", "not-there", "edit:0", [1.0, 0, 0], 1.0)
    sub, _ = make_sub(tmp_path / "b", n=5, extra=[orphan])
    with pytest.raises(ValueError, match="test split"):
        lb.validate(sub)
    other = LAB.loc["sentiment_hard"].index[10]
    sub, _ = make_sub(tmp_path / "c", n=5, extra=[("sentiment_hard", other, "edit:0", [1.0, 0, 0], 1.0)])
    with pytest.raises(ValueError, match="without an orig"):
        lb.validate(sub)
    sub, _ = make_sub(tmp_path / "d", n=5)
    (sub / "manifest.json").rename(tmp_path / "real.json")
    (sub / "manifest.json").symlink_to(tmp_path / "real.json")
    with pytest.raises(ValueError, match="symlink"):
        lb.validate(sub)


def test_build_ranks_complete_by_brier(tmp_path):
    root = tmp_path / "subs"
    y = LAB.loc["sentiment_hard"].label.to_numpy()
    good = [[0.9 if j == t else 0.05 for j in range(3)] for t in y]
    make_sub(root / "good", probs=good, manifest={"model": "org/best"})
    make_sub(root / "ok")
    make_sub(root / "tiny", n=10, manifest={"model": "org/tiny"})
    page = lb.build(root, tmp_path / "site").read_text()
    table = page.split("<h2>sentiment_hard</h2>")[1].split("</table>")[0]
    assert table.index("org/best") < table.index("org/m") < table.index("org/tiny")
    assert "partial, unranked" in table.split("org/tiny")[1]
    data = json.loads((tmp_path / "site" / "leaderboard.json").read_text())
    assert sorted(d["name"] for d in data) == ["good", "ok", "tiny"]


def test_render_escapes_manifest_text(tmp_path):
    sub, _ = make_sub(tmp_path / "m", n=5)
    s = lb.score(sub)
    s["model"], s["runner"] = "<script>alert(1)</script>", "<b>x</b>"  # as if validation were bypassed
    page = lb.render([s])
    assert "<script>alert" not in page and "<b>x" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_build_with_no_submissions(tmp_path):
    page = lb.build(tmp_path / "missing", tmp_path / "site").read_text()
    assert page.count("No submissions yet.") == 5


@pytest.mark.skipif(not os.environ.get("CBENCH_NETWORK"), reason="set CBENCH_NETWORK=1 to download datasets")
def test_labels_file_matches_loaders(tmp_path):
    fresh = pd.read_parquet(lb.write_labels(tmp_path / "labels.parquet"))
    committed = pd.read_parquet(lb.LABELS)
    pd.testing.assert_frame_equal(fresh[["dataset", "id", "label"]], committed[["dataset", "id", "label"]])
    both = fresh.soft.notna()
    assert (both == committed.soft.notna()).all()
    np.testing.assert_allclose(np.concatenate(fresh.soft[both].to_numpy()),
                               np.concatenate(committed.soft[both].to_numpy()))
