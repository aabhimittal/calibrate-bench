"""python -m cbench.runners {d1,logprob} MODEL_ID REVISION DATASET [DATASET ...] [--limit N] [--out DIR]"""

import argparse

from cbench.datasets import SOURCES
from cbench.runners import D1, LogprobLLM, run, write

p = argparse.ArgumentParser()
p.add_argument("runner", choices=["d1", "logprob"])
p.add_argument("model")
p.add_argument("revision", help="full commit sha")
p.add_argument("datasets", nargs="+", choices=list(SOURCES))
p.add_argument("--split", default="test", choices=["calib", "test"])
p.add_argument("--limit", type=int)
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--device")
p.add_argument("--out", help="default: submissions/<model name>")
a = p.parse_args()

runner = (D1 if a.runner == "d1" else LogprobLLM)(a.model, a.revision, device=a.device)
out = a.out or f"submissions/{a.model.split('/')[-1]}"
for name in a.datasets:
    preds = run(runner, name, a.split, a.limit, a.batch_size)
    info = {"split": a.split, "limit": a.limit, "batch_size": a.batch_size}
    print(name, len(preds), "rows ->", write(out, preds, runner.manifest, info))
