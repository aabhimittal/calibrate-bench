"""python -m cbench.leaderboard ...

  labels                      regenerate labels.parquet from the datasets (downloads them)
  guard < PATHS               check a PR's changed paths (NUL-separated, `git diff -z --name-only`) touch
                              exactly one submission; prints its name
  check DIR                   validate and score one submission; prints a Markdown summary
  build ROOT OUT              score every submission under ROOT into OUT/{leaderboard.json,index.html}
"""

import argparse
import sys
from pathlib import Path

from cbench import leaderboard as lb

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)
sub.add_parser("labels")
sub.add_parser("guard")
sub.add_parser("check").add_argument("dir", type=Path)
b = sub.add_parser("build")
b.add_argument("root", type=Path)
b.add_argument("out", type=Path)
a = p.parse_args()

try:
    if a.cmd == "labels":
        print(lb.write_labels())
    elif a.cmd == "guard":
        # NUL-separated so a filename with spaces or newlines can't split into innocent-looking paths
        print(lb.guard([f for f in sys.stdin.buffer.read().decode().split("\0") if f]))
    elif a.cmd == "check":
        s = lb.score(a.dir)
        print(f"### {s['model']} ({s['runner']}, {s['revision']})\n")
        print("| dataset | n | complete | accuracy | ECE | Brier | AURC |\n|---|---:|---|---:|---:|---:|---:|")
        for ds, d in s["datasets"].items():
            print(f"| {ds} | {d['n']} | {d['complete']} | {d['accuracy']:.3f} | {d['ece'][0]:.3f} | "
                  f"{d['brier'][0]:.3f} | {d['aurc'][0]:.3f} |")
    else:
        print(lb.build(a.root, a.out))
except ValueError as e:
    sys.exit(f"submission rejected:\n{e}")
