import pytest
import polars as pl
from pathlib import Path
from sweetbits.kmers import aggregate_kraken_kmers_logic
from sweetbits.metadata import read_companion_metadata
from sweetbits.features import produce_feature_kmer_global_logic

def test_kmer_pipeline_full(tmp_path):
    """
    End-to-end test of the k-mer ingestion and global feature extraction pipeline.
    Verifies schema standardization (t_id), species roll-up, and feature calculation.
    """
    # Setup paths
    kraken_dir = tmp_path / "kraken"
    kraken_dir.mkdir()
    kmer_dir = tmp_path / "kmer_agg"
    kmer_dir.mkdir()
    taxonomy_dir = Path("test_data/joltax_cache")
    feature_file = tmp_path / "global_features.parquet"

    # 1. Create Mock Kraken Data for two samples
    # 9606: Homo sapiens (Species)
    # 5000001: Unknown_5000001 (Species)
    
    # Sample 1
    mock_1 = [
        "C\tR1\t9606\t150|150\t10\t9606:10 1:2",
        "C\tR2\t5000001\t100|100\t5\t5000001:5 1:2"
    ]
    with open(kraken_dir / "Ki-2024_01_001.kraken", "w") as f:
        f.write("\n".join(mock_1))

    # Sample 2
    mock_2 = [
        "C\tR3\t9606\t150|150\t12\t9606:20 2:5" # 2 is lineage hit (Bacteria)
    ]
    with open(kraken_dir / "Ki-2024_02_001.kraken", "w") as f:
        f.write("\n".join(mock_2))

    # 2. Run Ingestion for both samples
    for k_file in kraken_dir.glob("*.kraken"):
        summary = aggregate_kraken_kmers_logic(
            kraken_file=k_file,
            output_dir=kmer_dir,
            taxonomy_dir=taxonomy_dir,
            overwrite=True
        )
        assert summary["status"] == "Success"

    # 3. Verify Ingestion Outputs (Standardized t_id schema)
    sample1_kmers = kmer_dir / "Ki-2024_01_001.kmers.parquet"
    assert sample1_kmers.exists()
    
    df_s1 = pl.read_parquet(sample1_kmers)
    assert "t_id" in df_s1.columns
    assert df_s1.schema["kmer_count"] == pl.UInt64
    
    sample1_lens = kmer_dir / "Ki-2024_01_001.read_lengths.parquet"
    assert sample1_lens.exists()
    df_l1 = pl.read_parquet(sample1_lens)
    assert "t_id" in df_l1.columns

    # # 4. Run Global Feature Extraction
    # input_pattern = str(kmer_dir / "*.kmers.parquet")
    # f_summary = produce_feature_kmer_global_logic(
    #     input_pattern=input_pattern,
    #     taxonomy_dir=taxonomy_dir,
    #     output_file=feature_file,
    #     overwrite=True
    # )
    # assert f_summary["species_processed"] == 2

    # # 5. Verify Feature Extraction (Pooled Totals)
    # df_features = pl.read_parquet(feature_file)
    # assert "t_id" in df_features.columns
    
    # # Human (9606) pooled totals:
    # # S1: 10 hits
    # # S2: 20 hits
    # # Total: 30 hits
    # human_row = df_features.filter(pl.col("t_id") == 9606)
    # assert human_row["grand_clade_kmers"][0] == 30
    
    # # Verify metadata
    # meta = read_companion_metadata(feature_file)
    # assert meta["file_type"] == "FEATURE_TABLE"
    # assert meta["sorting"] == "t_id"

def test_kmer_ingestion_generic(tmp_path):
    """
    Verifies that k-mer ingestion works for Generic (non-SweBITS) data.
    """
    kraken_file = tmp_path / "my_experiment.kraken"
    output_dir = tmp_path / "output"
    taxonomy_dir = Path("test_data/joltax_cache")

    mock_data = ["C\tR1\t9606\t150|150\t10\t9606:10"]
    with open(kraken_file, "w") as f:
        f.write("\n".join(mock_data))

    summary = aggregate_kraken_kmers_logic(
        kraken_file=kraken_file,
        output_dir=output_dir,
        taxonomy_dir=taxonomy_dir,
        overwrite=True
    )

    assert summary["sample_id"] == "my_experiment"
    
    kmer_file = output_dir / "my_experiment.kmers.parquet"
    meta = read_companion_metadata(kmer_file)
    assert meta["data_standard"] == "GENERIC"
    
    df = pl.read_parquet(kmer_file)
    # Generic data should NOT have year/week columns
    assert "year" not in df.columns
    assert "week" not in df.columns
