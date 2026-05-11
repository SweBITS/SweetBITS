import pytest
import polars as pl
from pathlib import Path
from sweetbits.features import produce_feature_kmer_sample_logic, produce_feature_kmer_stability_logic
from sweetbits.kmers import aggregate_kraken_kmers_logic

@pytest.fixture
def kmer_sample_data(tmp_path):
    """
    Generates kmers.parquet files from the Universal Golden Dataset.
    """
    kraken_dir = Path("test_data/universal_golden/inputs")
    taxonomy_dir = Path("test_data/joltax_cache")
    output_dir = tmp_path / "kmer_agg"
    output_dir.mkdir()
    
    # Process just 3 samples to save time, but enough for testing stability
    samples = sorted(list(kraken_dir.glob("*.kraken")))[:3]
    
    for kraken_file in samples:
        aggregate_kraken_kmers_logic(
            kraken_file=kraken_file,
            output_dir=output_dir,
            taxonomy_dir=taxonomy_dir,
            overwrite=True
        )
    
    return {
        "pattern": str(output_dir / "*.kmers.parquet"),
        "taxonomy": taxonomy_dir,
        "sample_count": 3
    }

def test_produce_feature_kmer_sample(kmer_sample_data, tmp_path):
    """
    Integration test for the long-format kmer-sample feature engine.
    Ensures all expected columns are present, bounded correctly, and Top 3 are extracted.
    """
    output_file = tmp_path / "kmer_sample_features.parquet"
    
    summary = produce_feature_kmer_sample_logic(
        input_pattern=kmer_sample_data["pattern"],
        taxonomy_dir=kmer_sample_data["taxonomy"],
        output_file=output_file,
        overwrite=True
    )
    
    assert summary["records_processed"] > 0
    assert output_file.exists()
    
    df = pl.read_parquet(output_file)
    cols = df.columns
    
    # Verify core columns
    assert "sample_id" in cols
    assert "t_id" in cols
    assert "kmers_sample_clade_count" in cols
    assert "kmers_sample_misclassifiedVSclassified_ratio" in cols
    
    # Verify 0-1 bounds for ratios and safe nulls
    ratio_cols = [c for c in cols if c.endswith("_ratio") and "VS" in c]
    for col in ratio_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val is not None:
            assert min_val >= 0.0
        if max_val is not None:
            assert max_val <= 1.0
            
    # Verify Top 3
    assert "kmers_sample_misclassified_top3_taxids" in cols
    assert "kmers_sample_exclade_top3_names" in cols
    assert "kmers_sample_misclassified_top3_shares" in cols

def test_produce_feature_kmer_stability(kmer_sample_data, tmp_path):
    """
    Integration test for the stability feature engine.
    """
    sample_file = tmp_path / "kmer_sample_features.parquet"
    stability_file = tmp_path / "kmer_stability_features.csv"
    
    produce_feature_kmer_sample_logic(
        input_pattern=kmer_sample_data["pattern"],
        taxonomy_dir=kmer_sample_data["taxonomy"],
        output_file=sample_file,
        overwrite=True
    )
    
    summary = produce_feature_kmer_stability_logic(
        input_parquet=sample_file,
        output_file=stability_file,
        overwrite=True
    )
    
    assert summary["taxa_processed"] > 0
    assert stability_file.exists()
    
    df = pl.read_csv(stability_file)
    cols = df.columns
    
    assert "t_id" in cols
    assert "kmers_stability_occupancy_ratio" in cols
    
    # Verify bounded stability presence stats
    assert "kmers_stability_cladeVStotal_ratio_presence" in cols
    
    pres_val = df["kmers_stability_cladeVStotal_ratio_presence"].max()
    if pres_val is not None:
        assert pres_val <= 1.0

    # Ensure unbounded rogue ratio was actually removed
    assert "kmers_sample_supportingVSmisclassified_ratio" not in cols
    assert "kmers_stability_supportingVSmisclassified_ratio_mean" not in cols
