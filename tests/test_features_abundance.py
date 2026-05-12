import polars as pl
import pytest
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
        
        # We allow small numerical differences due to floating point precision
        # and different quantile interpolation if any (though we tried to match it)
        gen_vals = df_gen[col].to_numpy()
        gold_vals = df_gold[col].to_numpy()
        
        import numpy as np
        # Golden data was saved with 6 decimal places, so we use a matching tolerance
        np.testing.assert_allclose(gen_vals, gold_vals, rtol=1e-4, atol=1e-5)

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
    assert "abundance_global_mean" in df.columns
    assert df.height == 2
