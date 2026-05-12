import polars as pl
import pytest
import numpy as np
from pathlib import Path
from sweetbits.features import produce_feature_abundance_logic

def test_feature_abundance_equivalence(tmp_path):
    """
    Verifies that produce_feature_abundance_logic matches the golden ground truth.
    """
    input_table = Path("test_data/universal_golden/inputs/golden_abundance_table.csv")
    golden_file = Path("test_data/universal_golden/ground_truth/golden_abundance_features.csv")
    output_file = tmp_path / "test_abundance_features.parquet"

    # 1. Run the logic
    produce_feature_abundance_logic(
        input_table=input_table,
        output_file=output_file,
        overwrite=True
    )

    # 2. Load results
    df_gen = pl.read_parquet(output_file).sort("t_id")
    df_gold = pl.read_csv(golden_file).sort("t_id")

    # 3. Compare
    assert df_gen.height == df_gold.height

    for col in df_gold.columns:
        if col == "t_id":
            continue

        # Map old golden name to new name
        new_col = col.replace("abundance_", "abund_")

        assert new_col in df_gen.columns

        # We allow small numerical differences due to floating point precision
        # and different quantile interpolation if any (though we tried to match it)
        gen_vals = df_gen[new_col].to_numpy()
        gold_vals = df_gold[col].to_numpy()
        np.testing.assert_allclose(gen_vals, gold_vals, rtol=5e-4, atol=1e-6)

    # Check for the new stdev column
    assert "abund_global_stdev" in df_gen.columns


def test_feature_abundance_tsv(tmp_path):
    """
    Verifies that the engine handles TSV input/output correctly.
    """
    # Create a dummy TSV
    input_tsv = tmp_path / "input.tsv"
    pl.DataFrame({
        "t_id": [1, 2],
        "S1": [1.0, 2.0],
        "S2": [3.0, 4.0]
    }).write_csv(input_tsv, separator="\t")

    output_tsv = tmp_path / "output.tsv"

    produce_feature_abundance_logic(
        input_table=input_tsv,
        output_file=output_tsv,
        overwrite=True
    )

    assert output_tsv.exists()
    df = pl.read_csv(output_tsv, separator="\t")
    assert "abund_global_mean" in df.columns
    assert "abund_global_stdev" in df.columns


def test_feature_abundance_minimizer_normalization(tmp_path):
    """
    Verifies that minimizer normalization is calculated correctly.
    """
    input_csv = tmp_path / "input.csv"
    pl.DataFrame({
        "t_id": [100, 200],
        "S1": [10.0, 50.0],
        "S2": [20.0, 150.0]
    }).write_csv(input_csv)

    inspect_csv = tmp_path / "inspect.csv"
    pl.DataFrame({
        "tax_id": [100, 200],
        "clade_minimizers": [10, 100]
    }).write_csv(inspect_csv)

    output_parquet = tmp_path / "output.parquet"

    produce_feature_abundance_logic(
        input_table=input_csv,
        output_file=output_parquet,
        inspect_file=inspect_csv,
        overwrite=True
    )

    df = pl.read_parquet(output_parquet).sort("t_id")

    # Taxon 100: mean abundance = 15.0, minimizers = 10 -> ratio = 1.5
    # Taxon 200: mean abundance = 100.0, minimizers = 100 -> ratio = 1.0

    row100 = df.filter(pl.col("t_id") == 100)
    assert row100["abund_global_meanVSmm_ratio"][0] == 1.5
    assert row100["abund_global_medianVSmm_ratio"][0] == 1.5

    row200 = df.filter(pl.col("t_id") == 200)
    assert row200["abund_global_meanVSmm_ratio"][0] == 1.0
    assert row200["abund_global_medianVSmm_ratio"][0] == 1.0
    assert df.height == 2
