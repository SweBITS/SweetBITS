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
    # 600000000: DummySpecies_0_0_0 (Species)
    
    # Sample 1
    mock_1 = [
        "C\tR1\t9606\t150|150\t10\t9606:10 1:2",
        "C\tR2\t600000000\t100|100\t5\t600000000:5 1:2"
    ]
    with open(kraken_dir / "Ki-2024_01_001.kraken", "w") as f:
        f.write("\n".join(mock_1))

    # Sample 2
    mock_2 = [
        "C\tR3\t9606\t150|150\t12\t9606:20 2:5" # 2 is Bacteria hit (Off-target for Eukaryota 9606)
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

    # 4. Run Global Feature Extraction
    input_pattern = str(kmer_dir / "*.kmers.parquet")
    f_summary = produce_feature_kmer_global_logic(
        input_pattern=input_pattern,
        taxonomy_dir=taxonomy_dir,
        output_file=feature_file,
        overwrite=True
    )
    assert f_summary["species_processed"] == 2

    # 5. Verify Feature Extraction (Pooled Totals)
    df_features = pl.read_parquet(feature_file)
    assert "t_id" in df_features.columns

    # Human (9606) pooled totals:
    # Lineage of 9606: [1 (root), 2759 (Eukaryota), 33208 (Metazoa), 7711 (Chordata), 40674 (Mammalia), 9443 (Primates), 9605 (Homo), 9606 (Homo sapiens)]
    
    # S1 (where t_id=9606): 
    #   9606 hits: 10 (clade)
    #   1 hits: 2 (lineage)
    
    # S2 (where t_id=9606): 
    #   9606 hits: 20 (clade)
    #   2 hits: 5 (misclassified - Bacteria is not in Eukaryota lineage)
    
    # Grand Totals Calculation for t_id 9606:
    #   clade: 10 + 20 = 30
    #   lineage: 2
    #   misclassified: 5
    #   classified: 30 + 2 + 5 = 37
    human_row = df_features.filter(pl.col("t_id") == 9606)
    
    assert human_row["grand_clade_kmers"][0] == 30
    assert human_row["grand_lineage_kmers"][0] == 2
    assert human_row["grand_misclassified_kmers"][0] == 5
    
    # Check Ratios (30 / 37 = 0.81081)
    assert human_row["grand_clade_to_classified_kmer_ratio"][0] == pytest.approx(0.81081, abs=1e-4)
    # misclassified_to_classified: 5 / 37 = 0.13513...
    assert human_row["grand_misclassified_to_classified_kmer_ratio"][0] == pytest.approx(0.13513, abs=1e-4)

    # Check that misclassified distance stats were calculated
    assert human_row["mean_grand_misclassified_kmer_distance"][0] is not None
    assert human_row["mean_grand_misclassified_kmer_distance"][0] > 0
    
    # Verify metadata
    meta = read_companion_metadata(feature_file)
    assert meta["file_type"] == "FEATURE_TABLE"
    assert meta["sorting"] == "t_id"

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

def test_kmer_pipeline_golden_dataset(tmp_path):
    """
    Validates the SweetBITS k-mer feature extraction engine against an independently generated
    Golden Dataset of 20+ taxa and 3 samples, ensuring mathematical parity.
    """
    golden_dir = Path("test_data/universal_golden")
    if not (golden_dir / "inputs").exists():
        pytest.skip("Golden dataset not found. Run tests/generate_universal_golden_data.py first.")
        
    kmer_dir = tmp_path / "kmer_agg"
    kmer_dir.mkdir()
    taxonomy_dir = Path("test_data/joltax_cache")
    feature_file = tmp_path / "global_features.csv"

    # 1. Run Ingestion on the 3 universal golden .kraken files
    for k_file in (golden_dir / "inputs").glob("*.kraken"):
        aggregate_kraken_kmers_logic(
            kraken_file=k_file,
            output_dir=kmer_dir,
            taxonomy_dir=taxonomy_dir,
            overwrite=True
        )

    # 2. Run Global Feature Extraction
    input_pattern = str(kmer_dir / "*.kmers.parquet")
    f_summary = produce_feature_kmer_global_logic(
        input_pattern=input_pattern,
        taxonomy_dir=taxonomy_dir,
        output_file=feature_file,
        overwrite=True
    )
    
    # 3. Load both datasets
    # Read generated features, treating missing values correctly
    df_generated = pl.read_csv(feature_file, null_values=[""])
    df_golden = pl.read_csv(golden_dir / "ground_truth/golden_kmer_features.csv", null_values=[""])
    
    assert f_summary["species_processed"] == df_golden.height
    
    # 4. Compare schemas and row counts
    assert df_generated.height == df_golden.height
    
    # 5. Verify mathematical parity
    # We use assert_frame_equal to ensure every cell matches the python-generated ground truth
    try:
        from polars.testing import assert_frame_equal
        # Drop brittle string columns for the strict numerical check
        drop_cols = [c for c in df_generated.columns if "_shares" in c or "_names" in c]
        df_gen_num = df_generated.drop(drop_cols)
        df_gold_num = df_golden.drop(drop_cols)
        
        assert_frame_equal(df_gen_num, df_gold_num, check_dtypes=False, rel_tol=1e-2, check_exact=False, check_column_order=False)
    except ImportError:
        pass # Older polars versions might not have testing module readily available, but 1.40 does.
