import polars as pl
import numpy as np
from pathlib import Path
from sweetbits.math import calc_clade_sum
from joltax import JolTree

tax_dir = Path("test_data/joltax_cache")
tree = JolTree.load(str(tax_dir))

df = pl.DataFrame({
    "sample_id": ["S1", "S1", "S1"],
    "t_id": [9606, 2, 1],
    "taxon_reads": [10, 20, 0]
})

result, synth = calc_clade_sum(df, tree, min_reads=0, min_observed=0)
res_dict = {row["t_id"]: row["clade_reads"] for row in result.to_dicts()}

for tid in [9606, 2759, 131567, 2, 1]:
    print(f"TaxID {tid}: {res_dict.get(tid)}")
