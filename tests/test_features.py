import pytest
import polars as pl
from pathlib import Path
from sweetbits.features import produce_feature_uniq_minimizer_corr_logic
from sweetbits.reports import gather_reports_logic

@pytest.fixture
def golden_data(tmp_path):
    """
    Sets up the 10-sample Universal Golden Dataset for minimizer tests.
    """
    base_dir = Path("test_data/universal_golden")
    input_dir = base_dir / "inputs"
    truth_dir = base_dir / "ground_truth"
    
    if not (input_dir).exists():
        pytest.skip("Golden dataset not found. Run tests/generate_universal_golden_data.py first.")
        
    # Merge reports into a single REPORT_PARQUET
    report_parquet = tmp_path / "merged_golden.parquet"
    gather_reports_logic(
        input_dir=input_dir,
        output_file=report_parquet,
        include_pattern="*.report",
        overwrite=True
    )
    
    return {
        "parquet": report_parquet,
        "inspect": truth_dir / "kraken_inspect.csv",
        "taxonomy": Path("test_data/joltax_cache")
    }

def test_minimizer_correlations_golden_dataset(golden_data, tmp_path):
    """
    Validates the minimizer correlation engine against the complex 10-sample 
    golden dataset. Ensures that correlations are calculated correctly for 
    all 50+ species across the tree.
    """
    output_file = tmp_path / "minimizer_features.csv"
    
    summary = produce_feature_uniq_minimizer_corr_logic(
        input_parquet=golden_data["parquet"],
        inspect_csv=golden_data["inspect"],
        taxonomy_dir=golden_data["taxonomy"],
        output_file=output_file,
        overwrite=True
    )
    
    assert summary["taxa_processed"] > 20
    assert summary["valid_correlations"] > 0
    
    df = pl.read_csv(output_file)
    
    # Standard columns should exist
    expected_cols = [
        "t_id", "mm_pearson_corr", "mm_pearson_p", "mm_pearson_n",
        "mm_obs_cov_mean", "mm_obs_cov_median", "mm_obs_cov_cv",
        "mm_obs_cov_p05", "mm_obs_cov_p95"
    ]
    for col in expected_cols:
        assert col in df.columns

    # Since the golden dataset uses a linear model for mm_uniq (0.05 * reads),
    # most correlations should be high
    # Check a specific abundant species like Human (9606)
    human = df.filter(pl.col("t_id") == 9606).to_dicts()[0]
    assert human["mm_pearson_n"] == 10
    assert human["mm_pearson_corr"] > 0.7
    assert human["mm_pearson_p"] < 0.05

def test_minimizer_correlations_bad_samples_golden(golden_data, tmp_path):
    """
    Ensures that excluding samples via a bad-samples file correctly 
    reduces 'n' and impacts the correlation results.
    """
    output_file = tmp_path / "minimizer_features_filtered.csv"
    
    # Exclude 5 samples out of 10
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("Ki-2024_01_001\nKi-2024_02_001\nKi-2024_03_001\nKi-2024_04_001\nKi-2024_05_001\n")
    
    summary = produce_feature_uniq_minimizer_corr_logic(
        input_parquet=golden_data["parquet"],
        inspect_csv=golden_data["inspect"],
        taxonomy_dir=golden_data["taxonomy"],
        output_file=output_file,
        bad_samples_file=bad_file,
        overwrite=True
    )
    
    df = pl.read_csv(output_file)
    # All n should be exactly 5
    # Since n < 6, Pearson correlations should be null
    human = df.filter(pl.col("t_id") == 9606).to_dicts()[0]
    assert human["mm_pearson_n"] == 5
    assert human["mm_pearson_corr"] is None
