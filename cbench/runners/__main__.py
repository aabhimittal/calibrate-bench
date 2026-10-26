"""python -m cbench.runners {d1,logprob,verbal} MODEL [REVISION] DATASET [DATASET ...] [--perturb FAMILY] [--out DIR]"""

import argparse

from cbench.datasets import SOURCES
from cbench.runners import D1, PERTURBATIONS, LogprobLLM, Verbal, run, write

p = argparse.ArgumentParser()
p.add_argument("runner", choices=["d1", "logprob", "verbal"])
p.add_argument("model")
p.add_argument("args", nargs="+", metavar="[REVISION] DATASET", help="full commit sha (not for verbal), then datasets")
p.add_argument("--split", default="test", choices=["calib", "test"])
p.add_argument("--limit", type=int)
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--perturb", choices=PERTURBATIONS, help="run one perturbation family instead of the originals")
p.add_argument("--variants", type=int, default=3, help="orders / edits / paraphrases per item")
p.add_argument("--device")
p.add_argument("--effort", default="low", help="verbal only")
p.add_argument("--weights", help="d1 only: REPO@SHA of a same-architecture checkpoint to run through d1's readout")
p.add_argument("--out", help="default: submissions/<model name>")
a = p.parse_args()

if a.runner == "verbal":
    runner, datasets = Verbal(a.model, a.effort), a.args
else:
    revision, *datasets = a.args
    runner = (D1(a.model, revision, a.device, a.weights) if a.runner == "d1"
              else LogprobLLM(a.model, revision, device=a.device))
if unknown := set(datasets) - set(SOURCES):
    p.error(f"unknown datasets {sorted(unknown)}; choose from {list(SOURCES)}")
out = a.out or f"submissions/{a.model.split('/')[-1]}"
for name in datasets:
    preds = run(runner, name, a.split, a.limit, a.batch_size, a.perturb, a.variants)
    info = {"split": a.split, "limit": a.limit, "batch_size": a.batch_size, "variants": a.variants if a.perturb else None}
    print(name, a.perturb or "orig", len(preds), "rows ->", write(out, preds, runner.manifest, info))
