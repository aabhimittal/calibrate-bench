"""Tables for FINDINGS.md: pip install -e . && python findings/analyze.py SUBMISSION_DIR [...]

Everything is recomputed from predictions.parquet with the same code the leaderboard uses.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cbench import leaderboard as lb
from cbench.metrics import brier, ece, reliability
from cbench.perturb import report

LAB = lb.labels()


def orig(sub: Path) -> pd.DataFrame:
    p = pd.read_parquet(sub / "predictions.parquet")
    return p[p.variant == "orig"]


def base_rate(ds: str, ids) -> np.ndarray:
    """Predict every item with the full test split's class frequencies: a prior a deployer would know, no model.

    Frequencies of the scored sample itself would leak its labels into the baseline.
    """
    full = LAB.loc[ds].label.to_numpy()
    freq = np.bincount(full, minlength=full.max() + 1) / len(full)
    return np.tile(freq, (len(ids), 1))


def selective_accuracy(p, y, coverage) -> float:
    """Accuracy on the most confident `coverage` share of items (ties broken by input order)."""
    keep = np.argsort(-p.max(1), kind="stable")[: int(round(coverage * len(p)))]
    return float((p[keep].argmax(1) == y[keep]).mean())


def paired_brier_diff(pa, pb, y, n=2000, seed=0):
    """Brier(a) - Brier(b) on the same items with a paired bootstrap 95% interval."""
    da = ((pa - np.eye(pa.shape[1])[y]) ** 2).sum(1)
    db = ((pb - np.eye(pb.shape[1])[y]) ** 2).sum(1)
    d = da - db
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    lo, hi = np.quantile(d[idx].mean(1), [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def main(subs: list[Path]):
    scored = {s.name: lb.score(s) for s in subs}
    datasets = sorted({ds for s in scored.values() for ds in s["datasets"]})

    print("## Calibration\n")
    print("| dataset | model | n | accuracy | ECE [95% CI] | Brier [95% CI] | soft ECE | soft Brier | "
          "acc @ top 50% | acc @ top 80% |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ds in datasets:
        rows = []
        for name, s in scored.items():
            if ds not in s["datasets"]:
                continue
            d, o = s["datasets"][ds], orig(subs[[x.name for x in subs].index(name)])
            o = o[o.dataset == ds]
            p, y = np.stack(o.probs.to_numpy()), LAB.loc[ds].loc[o.id].label.to_numpy()
            rows.append((name, d, p, y, o.id))
        for name, d, p, y, _ in rows:
            soft = (f"{d['soft_ece'][0]:.3f} | {d['soft_brier'][0]:.3f}" if "soft_ece" in d else "— | —")
            print(f"| {ds} | {name} | {d['n']} | {d['accuracy']:.3f} | {d['ece'][0]:.3f} [{d['ece'][1]:.3f}, "
                  f"{d['ece'][2]:.3f}] | {d['brier'][0]:.3f} [{d['brier'][1]:.3f}, {d['brier'][2]:.3f}] | {soft} | "
                  f"{selective_accuracy(p, y, .5):.3f} | {selective_accuracy(p, y, .8):.3f} |")
        # the base-rate predictor on the largest item set scored for this dataset
        name, d, p, y, ids = max(rows, key=lambda r: r[1]["n"])
        b = base_rate(ds, ids)
        print(f"| {ds} | *base rate* | {len(y)} | {(b.argmax(1) == y).mean():.3f} | {ece(b, y):.3f} | "
              f"{brier(b, y):.3f} | | | | |")

    print("\n## Paired comparisons (Brier difference, same items; negative = first model better)\n")
    names = list(scored)
    for ds in datasets:
        have = [n for n in names if ds in scored[n]["datasets"]]
        for a in have:  # every model against the base-rate predictor on its own items
            o = orig(subs[names.index(a)])
            o = o[o.dataset == ds]
            p, y = np.stack(o.probs.to_numpy()), LAB.loc[ds].loc[o.id].label.to_numpy()
            m, lo, hi = paired_brier_diff(p, base_rate(ds, o.id), y)
            print(f"- {ds}: {a} − base rate = {m:+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(y)} items")
        for i, a in enumerate(have):
            for bname in have[i + 1:]:
                oa, ob = (orig(subs[names.index(n)]) for n in (a, bname))
                oa, ob = oa[oa.dataset == ds].set_index("id"), ob[ob.dataset == ds].set_index("id")
                common = oa.index.intersection(ob.index)
                pa, pb = (np.stack(o.loc[common].probs.to_numpy()) for o in (oa, ob))
                y = LAB.loc[ds].loc[common].label.to_numpy()
                m, lo, hi = paired_brier_diff(pa, pb, y)
                print(f"- {ds}: {a} − {bname} = {m:+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(common)} items")

    print("\n## Reliability (equal-mass bins: mean confidence → accuracy, n)\n")
    for ds in datasets:
        for name in names:
            if ds not in scored[name]["datasets"]:
                continue
            o = orig(subs[names.index(name)])
            o = o[o.dataset == ds]
            p, y = np.stack(o.probs.to_numpy()), LAB.loc[ds].loc[o.id].label.to_numpy()
            conf, acc, size = reliability(p, y, n_bins=5)
            cells = ", ".join(f"{c:.2f}→{a:.2f} ({n})" for c, a, n in zip(conf, acc, size))
            print(f"- {ds} / {name}: {cells}")

    print("\n## Consistency\n")
    print("| model | dataset | family | items | flip rate | KL | any flip |\n|---|---|---|---:|---:|---:|---:|")
    for s in subs:
        r = report(pd.read_parquet(s / "predictions.parquet"))
        for _, x in r.iterrows():
            print(f"| {s.name} | {x.dataset} | {x.family} | {x['items']} | {x.flip_rate:.3f} | {x.kl:.3f} | "
                  f"{x.any_flip:.3f} |")

    print("\n## Base rates on the full test splits\n")
    for ds in LAB.index.get_level_values(0).unique():
        y = LAB.loc[ds].label.to_numpy()
        top = np.bincount(y).max() / len(y)
        print(f"- {ds}: majority class = {top:.3f} of {len(y)}")



if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]])
