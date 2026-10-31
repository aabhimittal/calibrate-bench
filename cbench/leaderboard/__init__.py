"""CI-verified submissions and the static leaderboard (M4).

A submission is submissions/<name>/{predictions.parquet, manifest.json} as written by cbench.runners.
Scores are recomputed here from the raw probabilities; a submission never carries its own numbers.
Labels come from labels.parquet (hashed ids, labels and rater distributions of every test split, no texts),
so CI needs no dataset downloads; a network test checks it against the loaders.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cbench.datasets import SOURCES
from cbench.metrics import _check, aurc, bootstrap_ci, brier, ece, reliability, risk_coverage

LABELS = Path(__file__).parent / "labels.parquet"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
VARIANT = re.compile(r"orig|(order|edit|attack|paraphrase):\d{1,3}")
IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,127}")  # model ids, commit shas, "api"
RUNNERS = {"d1", "logprob", "verbal"}
MAX_BYTES = 64 * 1024 * 1024
FILES = {"predictions.parquet", "manifest.json"}


def write_labels(path: Path = LABELS) -> Path:
    """Regenerate labels.parquet from the dataset loaders (downloads every dataset)."""
    from cbench.datasets import load

    rows = []
    for name in SOURCES:
        df = load(name).df
        df = df[df.split == "test"]
        soft = df.soft if "soft" in df else [None] * len(df)
        rows += [(name, i, int(y), None if s is None else list(s)) for i, y, s in zip(df.id, df.label, soft)]
    pd.DataFrame(rows, columns=["dataset", "id", "label", "soft"]).to_parquet(path, index=False)
    return path


def labels() -> pd.DataFrame:
    return pd.read_parquet(LABELS).set_index(["dataset", "id"])


def guard(changed: list[str]) -> str:
    """The one submission a PR may add or change; raises if it touches anything else.

    The scoring code must be the base branch's, so a submission PR can't also edit how it is scored.
    """
    dirs = set()
    for path in changed:
        parts = path.split("/")
        if len(parts) != 3 or parts[0] != "submissions" or not NAME.fullmatch(parts[1]) or parts[2] not in FILES:
            raise ValueError(f"{path}: a submission PR may only change submissions/<name>/{{{','.join(sorted(FILES))}}}")
        dirs.add(parts[1])
    if len(dirs) != 1:
        raise ValueError(f"a submission PR must change exactly one submission, got {sorted(dirs) or 'none'}")
    return dirs.pop()


def validate(sub: Path) -> tuple[dict, pd.DataFrame]:
    """Manifest and predictions of one submission, or ValueError listing every problem found."""
    problems = []
    if not NAME.fullmatch(sub.name):
        problems.append(f"directory name {sub.name!r} must match {NAME.pattern}")
    for f in FILES:
        if (sub / f).is_symlink():
            problems.append(f"{f} is a symlink")
        elif not (sub / f).is_file():
            problems.append(f"missing {f}")
        elif (sub / f).stat().st_size > MAX_BYTES:
            problems.append(f"{f} is larger than {MAX_BYTES >> 20} MB")
    if problems:
        raise ValueError("\n".join(problems))
    m = json.loads((sub / "manifest.json").read_text())
    for key in ("runner", "model", "revision", "readout", "runs"):
        if not m.get(key):
            problems.append(f"manifest.json has no {key!r}")
    # These end up in the CI summary (Markdown) and on the page: a newline or "|" could forge score rows
    if m.get("runner") not in RUNNERS:
        problems.append(f"manifest.json runner must be one of {sorted(RUNNERS)}")
    for key in ("model", "revision"):
        if not IDENT.fullmatch(str(m.get(key, ""))):
            problems.append(f"manifest.json {key} must match {IDENT.pattern}")
    price = [m.get("usd_per_hour")] if "usd_per_hour" in m else []
    if "usd_per_mtok" in m:
        mtok = m["usd_per_mtok"]
        price += [mtok.get("input"), mtok.get("output")] if isinstance(mtok, dict) else [None]
    if any(type(x) not in (int, float) or not 0 <= x < 1e6 for x in price):
        problems.append("manifest.json usd_per_hour / usd_per_mtok {input, output} must be non-negative numbers")
    preds = pd.read_parquet(sub / "predictions.parquet")
    want = ["dataset", "id", "variant", "probs", "latency_ms"]
    if list(preds.columns) != want:
        raise ValueError("\n".join(problems + [f"predictions.parquet columns must be {want}, got {list(preds.columns)}"]))
    lab = labels()
    if bad := sorted(set(preds.dataset) - set(SOURCES)):
        problems.append(f"unknown datasets {bad}")
    if bad := sorted(v for v in set(preds.variant) if not VARIANT.fullmatch(str(v))):
        problems.append(f"unknown variants {bad[:5]}")
    if preds.duplicated(["dataset", "id", "variant"]).any():
        problems.append("duplicate (dataset, id, variant) rows")
    for ds, g in preds[preds.dataset.isin(list(SOURCES))].groupby("dataset"):
        if missing := sorted(set(g.id) - set(lab.loc[ds].index))[:3]:
            problems.append(f"{ds}: ids not in the test split, e.g. {missing}")
            continue
        try:
            _check(g.probs.tolist(), lab.loc[ds].loc[g.id].label.to_numpy())
        except ValueError as e:
            problems.append(f"{ds}: {e}")
        if orphans := sorted(set(g[g.variant != "orig"].id) - set(g[g.variant == "orig"].id))[:3]:
            problems.append(f"{ds}: perturbed items without an orig prediction, e.g. {orphans}")
        if ds not in m.get("runs", {}):
            problems.append(f"{ds}: manifest.json has no runs entry")
        elif "usage" in m["runs"][ds]:
            u = m["runs"][ds]["usage"]
            if not isinstance(u, dict) or any(type(u.get(k)) is not int or u[k] < 0
                                              for k in ("input_tokens", "output_tokens")):
                problems.append(f"{ds}: runs usage must hold non-negative integer input_tokens and output_tokens")
    if problems:
        raise ValueError("\n".join(problems))
    return m, preds


def score(sub: Path) -> dict:
    """Every leaderboard number for one submission, recomputed from its probabilities."""
    from cbench.perturb import report

    m, preds = validate(sub)
    lab = labels()
    result = {k: m.get(k) for k in ("runner", "model", "revision", "readout", "served_model", "effort", "device", "gpu")}
    result |= {"name": sub.name, "datasets": {}}
    consistency = report(preds)
    for ds, g in preds[preds.variant == "orig"].groupby("dataset"):
        truth = lab.loc[ds].loc[g.id]
        p, y = np.stack(g.probs.to_numpy()), truth.label.to_numpy()
        run = m["runs"][ds]
        entry = {
            "n": len(g),
            "complete": len(g) == len(lab.loc[ds]),
            "accuracy": float((p.argmax(1) == y).mean()),
            **{f.__name__: bootstrap_ci(f, p, y) for f in (ece, brier, aurc)},
            "reliability": dict(zip(("conf", "acc", "size"), (v.tolist() for v in reliability(p, y, n_bins=10)))),
            "risk_coverage": _sampled(*risk_coverage(p, y)),
            "batch_size": run.get("batch_size"),
            # items in a batch share one timing (wall time / batch size), so percentiles only mean something at batch 1
            "latency_ms_mean": float(g.latency_ms.mean()),
            **({"latency_ms_p50": float(np.percentile(g.latency_ms, 50)),
                "latency_ms_p99": float(np.percentile(g.latency_ms, 99))} if run.get("batch_size") == 1 else {}),
            "usd_per_1k": usd_per_1k(m, run, g.latency_ms.mean()),
            "date": run.get("date"),
        }
        if truth.soft.notna().all():
            soft = np.stack(truth.soft.to_numpy())
            entry |= {"soft_ece": bootstrap_ci(ece, p, soft), "soft_brier": bootstrap_ci(brier, p, soft)}
        rows = consistency[consistency.dataset == ds]
        entry["perturbations"] = {r.family: {"flip_rate": r.flip_rate, "kl": r.kl, "any_flip": r.any_flip,
                                             "items": int(r["items"])} for _, r in rows.iterrows()}
        result["datasets"][ds] = entry
    return result


def _sampled(coverage, risk, points: int = 20) -> dict:
    """The risk-coverage curve at about `points` evenly spaced coverages, enough to draw it."""
    idx = np.unique(np.searchsorted(coverage, np.linspace(0, 1, points + 1)[1:] - 1e-9).clip(0, len(coverage) - 1))
    return {"coverage": coverage[idx].tolist(), "risk": risk[idx].tolist()}


def usd_per_1k(m: dict, run: dict, latency_ms: float) -> float | None:
    """$ per 1000 decisions at the submitter's declared price: hardware time for local models, tokens for APIs.

    Hardware cost uses the amortised per-decision time of the run's own batch size.
    """
    if "usd_per_mtok" in m and "usage" in run:
        asked = run["n"] + run.get("refused", 0)  # refused calls are billed too
        mtok, u = m["usd_per_mtok"], run["usage"]
        return (u["input_tokens"] * mtok["input"] + u["output_tokens"] * mtok["output"]) / 1e6 / asked * 1000
    if "usd_per_hour" in m:
        return latency_ms / 3.6e6 * m["usd_per_hour"] * 1000
    return None


def build(root: Path, out: Path) -> Path:
    """Score every submission under root into out/leaderboard.json and out/index.html."""
    scored = [score(d) for d in sorted(root.iterdir()) if d.is_dir()] if root.is_dir() else []
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.json").write_text(json.dumps(scored, indent=2) + "\n")
    (out / "index.html").write_text(render(scored))
    return out / "index.html"


def _ci(v) -> str:
    return f"{v[0]:.3f} <small>[{v[1]:.3f}, {v[2]:.3f}]</small>" if v else "—"


def _latency(d) -> str:
    if "latency_ms_p50" in d:
        return f"p50 {d['latency_ms_p50']:.0f} / p99 {d['latency_ms_p99']:.0f}"
    return f"{d['latency_ms_mean']:.0f} <small>(mean, batch {html.escape(str(d['batch_size']))})</small>"


def _diagram(name: str, rel: dict) -> str:
    """Reliability diagram as inline SVG: one dot per equal-mass bin, the diagonal is perfect calibration."""
    side, pad = 180, 28
    x = lambda v: pad + v * (side - pad - 8)  # noqa: E731
    y = lambda v: side - pad - v * (side - pad - 8)  # noqa: E731
    pts = list(zip(rel["conf"], rel["acc"], rel["size"]))
    dots = "".join(f'<circle cx="{x(c):.1f}" cy="{y(a):.1f}" r="4"><title>confidence {c:.2f}, accuracy {a:.2f}, '
                   f'{n} items</title></circle>' for c, a, n in pts)
    line = " ".join(f"{x(c):.1f},{y(a):.1f}" for c, a, _ in pts)
    gap = sum(n * abs(c - a) for c, a, n in pts) / sum(n for *_, n in pts)
    return (f'<figure><svg viewBox="0 0 {side} {side}" width="{side}" height="{side}" role="img" '
            f'aria-label="Reliability diagram for {name}: mean confidence against accuracy in {len(pts)} '
            f'equal-mass bins, ECE {gap:.3f}">'
            f'<rect x="{x(0)}" y="{y(1)}" width="{x(1) - x(0)}" height="{y(0) - y(1)}" class="frame"/>'
            f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" class="ideal"/>'
            f'<polyline points="{line}" class="series"/>{dots}'
            f'<text x="{x(0.5)}" y="{side - 6}" text-anchor="middle">confidence</text>'
            f'<text transform="translate(12 {y(0.5)}) rotate(-90)" text-anchor="middle">accuracy</text>'
            f'<text x="{x(0)}" y="{y(0) + 13}" text-anchor="middle">0</text>'
            f'<text x="{x(1)}" y="{y(0) + 13}" text-anchor="middle">1</text>'
            f'</svg><figcaption>{name}</figcaption></figure>')


def _curve(name: str, rc: dict) -> str:
    """Risk-coverage curve as inline SVG: error rate among the most confident share of items. The dashed line is the
    overall error rate, what accepting items at random would give at every coverage."""
    side, pad = 180, 28
    top = max(0.05, float(np.ceil(max(rc["risk"]) * 10) / 10))
    x = lambda v: pad + v * (side - pad - 8)  # noqa: E731
    y = lambda v: side - pad - v / top * (side - pad - 8)  # noqa: E731
    line = " ".join(f"{x(c):.1f},{y(r):.1f}" for c, r in zip(rc["coverage"], rc["risk"]))
    dots = "".join(f'<circle cx="{x(c):.1f}" cy="{y(r):.1f}" r="3"><title>top {c:.0%} by confidence: error rate '
                   f'{r:.3f}</title></circle>' for c, r in zip(rc["coverage"], rc["risk"]))
    full = rc["risk"][-1]
    return (f'<figure><svg viewBox="0 0 {side} {side}" width="{side}" height="{side}" role="img" '
            f'aria-label="Risk-coverage curve for {name}: error rate {rc["risk"][0]:.3f} on the most confident '
            f'{rc["coverage"][0]:.0%}, {full:.3f} on all items">'
            f'<rect x="{x(0)}" y="{y(top)}" width="{x(1) - x(0)}" height="{y(0) - y(top)}" class="frame"/>'
            f'<line x1="{x(0)}" y1="{y(full):.1f}" x2="{x(1)}" y2="{y(full):.1f}" class="ideal"/>'
            f'<polyline points="{line}" class="series"/>{dots}'
            f'<text x="{x(0.5)}" y="{side - 6}" text-anchor="middle">coverage</text>'
            f'<text transform="translate(12 {y(top / 2)}) rotate(-90)" text-anchor="middle">error rate</text>'
            f'<text x="{x(0)}" y="{y(0) + 13}" text-anchor="middle">0</text>'
            f'<text x="{x(1)}" y="{y(0) + 13}" text-anchor="middle">1</text>'
            f'<text x="{x(0) - 4}" y="{y(top) + 4}" text-anchor="end">{top:g}</text>'
            f'</svg><figcaption>{name}: risk vs coverage</figcaption></figure>')


def render(scored: list[dict]) -> str:
    """Static page: one table per dataset, complete submissions ranked by Brier, partial ones listed after."""
    def e(v) -> str:  # model names and manifest text come from pull requests
        return html.escape(str(v))

    sections = []
    for ds, src in SOURCES.items():
        rows = [(s, s["datasets"][ds]) for s in scored if ds in s["datasets"]]
        rows.sort(key=lambda r: (not r[1]["complete"], r[1]["brier"][0]))
        soft = any("soft_ece" in d for _, d in rows)
        head = ["Model", "Readout", "n", "Accuracy", "ECE", "Brier", "AURC"]
        head += ["Soft ECE", "Soft Brier"] if soft else []
        head += ["Order flips", "Typo any-flip", "Attack success", "Latency (ms)", "$ / 1k decisions"]
        body = []
        for s, d in rows:
            pert = d["perturbations"]
            cells = [
                f"{e(s['model'])}<br><small>{e(str(s['revision'])[:12])}</small>",
                e(s["runner"]),
                f"{d['n']}" + ("" if d["complete"] else " <small>(partial, unranked)</small>"),
                f"{d['accuracy']:.3f}",
                _ci(d["ece"]), _ci(d["brier"]), _ci(d["aurc"]),
            ]
            cells += [_ci(d.get("soft_ece")), _ci(d.get("soft_brier"))] if soft else []
            cells += [
                f"{pert['order']['flip_rate']:.3f}" if "order" in pert else "—",
                f"{pert['edit']['any_flip']:.3f}" if "edit" in pert else "—",
                f"{pert['attack']['any_flip']:.3f}" if "attack" in pert else "—",
                _latency(d) + f"<br><small>{e(s.get('gpu') or s.get('device') or '?')}</small>",
                f"{d['usd_per_1k']:.4f}" if d.get("usd_per_1k") is not None else "—",
            ]
            body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        if not body:
            body = [f'<tr><td colspan="{len(head)}">No submissions yet.</td></tr>']
        figs = "".join(_diagram(e(s["model"]), d["reliability"]) + _curve(e(s["model"]), d["risk_coverage"])
                       for s, d in rows)
        sections.append(
            f"<section><h2>{e(ds)}</h2><p class=\"note\">{e(src.task)}. Lower ECE, Brier and AURC are better; "
            f"brackets are 95% bootstrap intervals.</p><div class=\"scroll\"><table aria-label=\"{e(ds)}\"><thead><tr>"
            + "".join(f'<th scope="col">{h}</th>' for h in head)
            + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
            + (f'<div class="diagrams">{figs}</div>' if figs else "") + "</section>"
        )
    return PAGE.format(sections="\n".join(sections))


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CALIBRATE-BENCH leaderboard</title>
<style>
:root {{ --bg: #fff; --fg: #1a1a1a; --muted: #5c5c5c; --line: #ddd; --head: #f4f4f4; --series: #2a78d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #141414; --fg: #eee; --muted: #aaa; --line: #333; --head: #1f1f1f;
  --series: #3987e5; }} }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; margin: 0 auto;
       max-width: 72rem; padding: 1.5rem 1rem; }}
small, .note {{ color: var(--muted); }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ border-bottom: 1px solid var(--line); padding: .4rem .6rem; text-align: left; vertical-align: top; }}
th {{ background: var(--head); }}
.diagrams {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 1rem; }}
figure {{ margin: 0; flex: 0 1 180px; min-width: 140px; }}
figcaption {{ color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }}
svg {{ max-width: 100%; height: auto; }}
svg text {{ fill: var(--muted); font-size: 11px; }}
svg .frame {{ fill: none; stroke: var(--line); }}
svg .ideal {{ stroke: var(--muted); stroke-dasharray: 4 4; }}
svg .series {{ fill: none; stroke: var(--series); stroke-width: 2; }}
svg circle {{ fill: var(--series); stroke: var(--bg); stroke-width: 2; }}
</style>
</head>
<body>
<h1>CALIBRATE-BENCH</h1>
<p>An independent audit of calibration in decision models and LLMs. Every number here is recomputed by CI from
submitted probability vectors; see the <a href="leaderboard.json">raw scores</a>.
Readouts measure different things: <code>d1</code> and <code>logprob</code> use model probabilities,
<code>verbal</code> uses probabilities the model states. Compare within a readout.
Soft scores compare against the share of raters picking each option; their floor is the raters' own disagreement.
Latency percentiles are shown only for runs at batch size 1; batched runs share one timing per batch, so they show the
amortised mean. Cost is hardware time (or API tokens) at the price each submitter declares, on the device shown.
Attack success: share of items whose answer a greedy search flips within its edit budget (typos and
contraction toggles such as "do not" / "don't"); lower is more robust.
Reliability diagrams: each dot is one tenth of the items by confidence; dots below the dashed diagonal are
overconfident. Risk-coverage curves: the error rate if only the most confident share of items is automated; the dashed
line is the overall error rate, which abstaining at random would give, so a useful curve starts well below it.</p>
{sections}
</body>
</html>
"""
