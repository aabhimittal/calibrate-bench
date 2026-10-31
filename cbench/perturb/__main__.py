"""python -m cbench.perturb ...

  paraphrase DATASET [--items N] [--k K] [--model M]   generate and store paraphrases (needs Anthropic credentials)
  spotcheck DATASET OUT.csv [--n N]                     sample pairs for a human to mark same_meaning y/n
  agreement SHEET.csv                                   share of checked pairs marked meaning-preserving
  lexicon                                               regenerate the real-word list the attack avoids (downloads)
  report PREDICTIONS.parquet                            flip rate, KL and attack success per dataset/perturbation
"""

import argparse

import pandas as pd

from cbench import perturb

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)
g = sub.add_parser("paraphrase")
g.add_argument("dataset")
g.add_argument("--items", type=int, default=200)
g.add_argument("--k", type=int, default=3)
g.add_argument("--model", default="claude-opus-5-5")
s = sub.add_parser("spotcheck")
s.add_argument("dataset")
s.add_argument("out")
s.add_argument("--n", type=int, default=50)
sub.add_parser("agreement").add_argument("sheet")
sub.add_parser("lexicon")
sub.add_parser("report").add_argument("predictions")
a = p.parse_args()

if a.cmd == "paraphrase":
    print(perturb.generate_paraphrases(a.dataset, a.items, a.k, a.model))
elif a.cmd == "spotcheck":
    perturb.spotcheck_sheet(a.dataset, a.n).to_csv(a.out, index=False)
    print(a.out)
elif a.cmd == "lexicon":
    print(perturb.write_lexicon())
elif a.cmd == "agreement":
    share, n = perturb.agreement(pd.read_csv(a.sheet, keep_default_na=False))
    print(f"{share:.1%} of {n} checked pairs keep the meaning")
else:
    print(perturb.report(pd.read_parquet(a.predictions)).to_string(index=False))
