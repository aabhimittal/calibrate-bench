"""python -m cbench.datasets [name ...] -- print the license manifest and load each dataset."""

import sys

from cbench.datasets import SOURCES, load

for name in sys.argv[1:] or SOURCES:
    src = SOURCES[name]
    if src.blocked:
        print(f"{name:18} BLOCKED  {src.blocked}")
        continue
    ds = load(name)
    counts = ds.df.split.value_counts().to_dict()
    print(f"{name:18} {src.license:10} {len(ds.options):>3} options  {counts}  rev={src.revision[:12]}")
