import pytest
import polars as pl
import numpy as np
from pathlib import Path
from sweetbits.math import calc_clade_sum
from joltax import JolTree

def test_calc_clade_sum_basic(tmp_path):
    tax_dir = Path("test_data/joltax_cache")
    if not tax_dir.exists():
        pytest.skip("JolTax cache not found")
        
    tree = JolTree.load(str(tax_dir))
    
    # Eukaryota (9606) has depth ~8, Bacteria (2) has depth 1. 
    # Root (1).
    # Let's use some IDs we know: 
    # 2 (Bacteria), 2759 (Eukaryota), 9606 (Homo sapiens), 1 (root)
    
    df = pl.DataFrame({
        "sample_id": ["S1", "S1", "S1"],
        "t_id": [9606, 2, 1],
        "taxon_reads": [10, 20, 0]
    })
    
    result, synth = calc_clade_sum(df, tree, min_reads=0, min_observed=0)
    
    res_dict = {row["t_id"]: row["clade_reads"] for row in result.to_dicts()}
    
    # 9606 clade should be 10
    assert res_dict[9606] == 10
    # 2 clade should be 20
    assert res_dict[2] == 20
    # 1 (root) should get votes from 9606 and 2 (so 10 + 20 = 30)
    assert res_dict[1] == 30
    
def test_calc_clade_sum_filtering(tmp_path):
    tax_dir = Path("test_data/joltax_cache")
    if not tax_dir.exists():
        pytest.skip("JolTax cache not found")
        
    tree = JolTree.load(str(tax_dir))
    
    df = pl.DataFrame({
        "sample_id": ["S1", "S1", "S1"],
        "t_id": [9606, 2, 1],
        "taxon_reads": [10, 20, 0]
    })
    
    # Set min_reads = 15. 9606 (10) should fail. 2 (20) should pass.
    # 1 (root) should only get vote from 2 (20).
    result, synth = calc_clade_sum(df, tree, min_reads=15, min_observed=1, keep_composition=True)
    
    res_dict = {row["t_id"]: row["clade_reads"] for row in result.to_dicts()}
    
    assert res_dict[9606] == 0  # Failed
    assert res_dict[2] == 20    # Passed
    assert res_dict[1] == 20    # Only got 20 from Bacteria
    
    assert synth[0] == 10.0 # 10 reads from 9606 went to synthetic bin
