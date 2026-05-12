import pytest
import polars as pl
from pathlib import Path
from sweetbits.features import produce_feature_kmer_sample_logic, produce_feature_kmer_stability_logic
from sweetbits.kmers import aggregate_kraken_kmers_logic
from polars.testing import assert_frame_equal

@pytest.fixture
def kmer_sample_data(tmp_path):
    kraken_dir = Path("test_data/universal_golden/inputs")
    taxonomy_dir = Path("test_data/joltax_cache")
    output_dir = tmp_path / "kmer_agg"
    output_dir.mkdir()
    
    # Process all 10 samples to match golden dataset exactly
    for kraken_file in kraken_dir.glob("*.kraken"):
        aggregate_kraken_kmers_logic(
            kraken_file=kraken_file,
            output_dir=output_dir,
            taxonomy_dir=taxonomy_dir,
            overwrite=True
        )
    
    return {
        "pattern": str(output_dir / "*.kmers.parquet"),
        "taxonomy": taxonomy_dir
    }

def test_produce_feature_kmer_sample_parity(kmer_sample_data, tmp_path):
    output_file = tmp_path / "kmer_sample_features.parquet"
    golden_file = Path("test_data/universal_golden/ground_truth/golden_kmer_sample_features.csv")
    
    produce_feature_kmer_sample_logic(
        input_pattern=kmer_sample_data["pattern"],
        taxonomy_dir=kmer_sample_data["taxonomy"],
        output_file=output_file,
        overwrite=True
    )
    
    df_gen = pl.read_parquet(output_file)
    df_gold = pl.read_csv(golden_file, null_values=[""])
    
    assert df_gen.height == df_gold.height, "Row counts differ between generated and golden sample features."
    
    # Drop string lists (names/ids/shares) to just compare numerical correctness
    drop_cols = [c for c in df_gen.columns if "_shares" in c or "_names" in c or "_taxids" in c]
    df_gen_num = df_gen.drop(drop_cols).sort(["sample_id", "t_id"])
    
    # Filter gold to just the taxa that were preserved in the generated output (species assigned)
    df_gen_num = df_gen_num.with_columns(pl.col("sample_id").cast(pl.String))
    df_gold = df_gold.with_columns(pl.col("sample_id").cast(pl.String))
    df_gold_filtered = df_gold.join(df_gen_num.select(["sample_id", "t_id"]), on=["sample_id", "t_id"], how="inner").sort(["sample_id", "t_id"])
    
    common_cols = sorted(list(set(df_gen_num.columns) & set(df_gold_filtered.columns)))
    
    assert_frame_equal(
        df_gen_num.select(common_cols),
        df_gold_filtered.select(common_cols),
        check_dtypes=False,
        rel_tol=1e-2,
        check_exact=False,
        check_column_order=False
    )

def test_produce_feature_kmer_stability_parity(kmer_sample_data, tmp_path):
    sample_file = tmp_path / "kmer_sample_features.parquet"
    stability_file = tmp_path / "kmer_stability_features.csv"
    golden_file = Path("test_data/universal_golden/ground_truth/golden_kmer_stability_features.csv")
    
    produce_feature_kmer_sample_logic(
        input_pattern=kmer_sample_data["pattern"],
        taxonomy_dir=kmer_sample_data["taxonomy"],
        output_file=sample_file,
        overwrite=True
    )
    
    produce_feature_kmer_stability_logic(
        input_parquet=sample_file,
        output_file=stability_file,
        overwrite=True
    )
    
    df_gen = pl.read_csv(stability_file, null_values=[""])
    df_gold = pl.read_csv(golden_file, null_values=[""])
    
    assert df_gen.height == df_gold.height, "Row counts differ between generated and golden stability features."
    
    df_gen_num = df_gen.sort("t_id")
    df_gold_filtered = df_gold.join(df_gen_num.select(["t_id"]), on="t_id", how="inner").sort("t_id")
    
    common_cols = sorted(list(set(df_gen_num.columns) & set(df_gold_filtered.columns)))
    
    assert_frame_equal(
        df_gen_num.select(common_cols), 
        df_gold_filtered.select(common_cols), 
        check_dtypes=False, 
        rel_tol=1e-2, 
        check_exact=False, 
        check_column_order=False
    )
