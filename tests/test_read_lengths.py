import pytest
import polars as pl
import numpy as np
from pathlib import Path
from sweetbits.features import (
    calculate_weighted_stats, 
    produce_feature_read_lengths_global_logic,
    produce_feature_read_lengths_sample_logic
)
from sweetbits.kmers import aggregate_kraken_kmers_logic

@pytest.fixture
def read_length_golden_data(tmp_path):
    """
    Generates read_lengths.parquet files from the Universal Golden Dataset.
    """
    kraken_dir = Path("test_data/universal_golden/inputs")
    taxonomy_dir = Path("test_data/joltax_cache")
    output_dir = tmp_path / "read_lengths"
    output_dir.mkdir()
    
    # Process just the first 3 samples to save time, but enough for testing
    samples = sorted(list(kraken_dir.glob("*.kraken")))[:3]
    
    for kraken_file in samples:
        aggregate_kraken_kmers_logic(
            kraken_file=kraken_file,
            output_dir=output_dir,
            taxonomy_dir=taxonomy_dir,
            overwrite=True
        )
    
    return {
        "pattern": str(output_dir / "*.read_lengths.parquet"),
        "taxonomy": taxonomy_dir
    }

def test_calculate_weighted_stats_multi_group():
    """
    Unit test for the helper function to ensure it handles multi-column grouping.
    """
    data = pl.DataFrame({
        "t_id": [100, 100, 200],
        "sample_id": ["S1", "S1", "S1"],
        "val": [10, 20, 100],
        "weight": [1, 1, 10]
    }).lazy()
    
    # Single group (existing behavior)
    res_single = calculate_weighted_stats(data, "val", "weight", "t_id", "test").collect()
    assert res_single.height == 2
    
    # Multi group (new behavior)
    res_multi = calculate_weighted_stats(data, "val", "weight", ["t_id", "sample_id"], "test").collect()
    assert res_multi.height == 2
    assert "sample_id" in res_multi.columns

def test_produce_feature_read_lengths_global(read_length_golden_data, tmp_path):
    """
    Integration test for the new 'read-lengths-global' feature engine.
    """
    global_out = tmp_path / "global_read_lengths.csv"
    
    summary = produce_feature_read_lengths_global_logic(
        input_pattern=read_length_golden_data["pattern"],
        output_file=global_out,
        min_reads=5, # Lower for test data
        overwrite=True
    )
    
    assert summary["taxa_processed"] > 0
    assert global_out.exists()
    
    # Verify Global Stats
    global_df = pl.read_csv(global_out)
    assert "reads_global_readlen_mean" in global_df.columns
    assert "reads_global_readlen_cv" in global_df.columns
    assert "reads_global_total_count" in global_df.columns
    
    # Verify that the companion metadata exists
    assert Path(str(global_out) + ".json").exists()

def test_produce_feature_read_lengths_sample(read_length_golden_data, tmp_path):
    """
    Integration test for the new 'read-lengths-sample' feature engine.
    """
    sample_out = tmp_path / "sample_read_lengths.parquet"
    
    summary = produce_feature_read_lengths_sample_logic(
        input_pattern=read_length_golden_data["pattern"],
        output_file=sample_out,
        overwrite=True
    )
    
    assert summary["records_processed"] > 0
    assert sample_out.exists()
    
    # Verify Sample Stats
    sample_df = pl.read_parquet(sample_out)
    assert "sample_id" in sample_df.columns
    assert "reads_sample_readlen_mean" in sample_df.columns
    assert "reads_sample_total_count" in sample_df.columns
    
    # Verify that the companion metadata exists
    assert Path(str(sample_out) + ".json").exists()

def test_calculate_weighted_stats_overflow_protection():
    """
    Ensures that calculate_weighted_stats does not overflow when handling 
    highly abundant taxa (e.g. 50M reads * 150bp).
    """
    # 50,000,000 * 150 = 7,500,000,000 (Exceeds UInt32 limit of 4.29B)
    data = pl.DataFrame({
        "t_id": [9606],
        "length": [150],
        "count": [50_000_000]
    }).with_columns([
        pl.col("t_id").cast(pl.UInt32),
        pl.col("length").cast(pl.UInt16),
        pl.col("count").cast(pl.UInt64)
    ])
    
    stats_lf = calculate_weighted_stats(
        data.lazy(),
        metric_col="length",
        weight_col="count",
        group_col="t_id",
        suffix="len"
    )
    
    result = stats_lf.collect().to_dicts()[0]
    
    # If overflow happened, mean would be incorrect (approx 0.74 due to wrap-around)
    assert result["len_mean"] == 150.0
    assert result["len_median"] == 150.0
