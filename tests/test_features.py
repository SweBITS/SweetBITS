import pytest
import polars as pl
import numpy as np
from pathlib import Path
from sweetbits.features import generate_minimizer_correlations
from joltax import JolTree

def test_minimizer_correlations_basic(tmp_path):
    tax_dir = Path("test_data/joltax_cache")
    if not tax_dir.exists():
        pytest.skip("JolTax cache not found")
        
    tree = JolTree.load(str(tax_dir))
    
    # Mock some data for a single TaxID (9606 - Homo sapiens) across 10 samples
    # We want a perfect correlation for S1-S10
    samples = [f"S{i}" for i in range(1, 11)]
    reads = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # mm_uniq should scale with reads for perfect correlation
    mm_uniq = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    mm_tot = [x * 2 for x in mm_uniq]
    
    df = pl.DataFrame({
        "sample_id": samples,
        "t_id": [9606] * 10,
        "taxon_reads": reads,
        "mm_tot": mm_tot,
        "mm_uniq": mm_uniq
    }).lazy()
    
    # Mock inspect data (Total minimizers in DB for 9606)
    # Let's say it has 100,000 minimizers in the DB.
    inspect_df = pl.DataFrame({
        "tax_id": [9606],
        "clade_minimizers": [100000]
    }).lazy()
    
    # Run engine
    result, long_df = generate_minimizer_correlations(df, inspect_df, tree)
    
    # It should have 9606, 2759, and 1 (ancestors added by calc_clade_sum)
    assert result.height >= 1
    
    # Check long-format columns
    expected_cols = {"t_id", "sample_id", "reads_tot", "mm_tot", "mm_uniq", 
                     "mm_uniq_per_read", "mm_uniq_prop", "mm_obs_cov", "mm_exp_cov"}
    assert set(long_df.columns) == expected_cols
    
    # Find the row for 9606
    row_9606 = result.filter(pl.col("t_id") == 9606).to_dicts()[0]
    
    assert row_9606["mm_pearson_n"] == 10
    # Perfect linear relationship should give R = 1.0
    assert pytest.approx(row_9606["mm_pearson_coeff"], 0.001) == 1.0
    # p-value should be very low
    assert row_9606["mm_pearson_p"] < 0.05
    
    # Distributional stats
    assert row_9606["mm_obs_cov_mean"] > 0
    assert row_9606["mm_obs_cov_cv"] > 0

def test_minimizer_correlations_safety_filter(tmp_path):
    tax_dir = Path("test_data/joltax_cache")
    if not tax_dir.exists():
        pytest.skip("JolTax cache not found")
        
    tree = JolTree.load(str(tax_dir))
    
    # Mock data with only 5 samples (below the n=6 safety limit)
    samples = [f"S{i}" for i in range(1, 6)]
    df = pl.DataFrame({
        "sample_id": samples,
        "t_id": [9606] * 5,
        "taxon_reads": [100] * 5,
        "mm_tot": [20] * 5,
        "mm_uniq": [10] * 5
    }).lazy()
    
    inspect_df = pl.DataFrame({
        "tax_id": [9606],
        "clade_minimizers": [100000]
    }).lazy()
    
    result, _ = generate_minimizer_correlations(df, inspect_df, tree)
    
    row = result.to_dicts()[0]
    # Pearson should be NaN/None due to n < 6
    assert row["mm_pearson_coeff"] is None
    assert row["mm_pearson_p"] is None
    assert row["mm_pearson_n"] == 5

def test_minimizer_correlations_bad_samples(tmp_path):
    tax_dir = Path("test_data/joltax_cache")
    if not tax_dir.exists():
        pytest.skip("JolTax cache not found")
        
    tree = JolTree.load(str(tax_dir))
    
    # 10 samples, but 5 are "bad"
    samples = [f"S{i}" for i in range(1, 11)]
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("S1\nS2\nS3\nS4\nS5\n")
    
    df = pl.DataFrame({
        "sample_id": samples,
        "t_id": [9606] * 10,
        "taxon_reads": [100] * 10,
        "mm_tot": [20] * 10,
        "mm_uniq": [10] * 10
    }).lazy()
    
    inspect_df = pl.DataFrame({
        "tax_id": [9606],
        "clade_minimizers": [100000]
    }).lazy()
    
    result, _ = generate_minimizer_correlations(df, inspect_df, tree, bad_samples_file=bad_file)
    
    row = result.to_dicts()[0]
    # Should only have 5 samples remaining, so pearson should be NaN
    assert row["mm_pearson_n"] == 5
    assert row["mm_pearson_coeff"] is None
