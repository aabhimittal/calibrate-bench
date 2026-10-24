"""python -m cbench.datasets [name ...] -- print the license manifest and load each dataset."""

import sys

from cbench.datasets import SOURCES, load

for name in sys.argv[1:] or SOURCES:
    src = SOURCES[name]
    ds = load(name)
    counts = ds.df.split.value_counts().to_dict()
    soft = " soft" if "soft" in ds.df else ""
    print(f"{name:18} {src.license:10} {len(ds.options):>3} options{soft:5}  {counts}  rev={src.revision[:12]}")
