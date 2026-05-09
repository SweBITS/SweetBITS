import pytest
import polars as pl
from pathlib import Path
from sweetbits.tables import generate_table_logic
from sweetbits.reports import gather_reports_logic
from polars.testing import assert_frame_equal

@pytest.fixture
def golden_data(tmp_path):
    """
    Sets up the Universal Golden Dataset for table tests.
    Returns paths to inputs and ground truth.
    """
    base_dir = Path("test_data/universal_golden")
    input_dir = base_dir / "inputs"
    truth_dir = base_dir / "ground_truth"
    
    if not input_dir.exists():
        pytest.skip("Golden dataset not found. Run tests/generate_universal_golden_data.py first.")
        
    # 1. Merge the 3 golden reports into a single REPORT_PARQUET
    report_parquet = tmp_path / "merged_golden.parquet"
    gather_reports_logic(
        input_dir=input_dir,
        output_file=report_parquet,
        include_pattern="*.report",
        overwrite=True
    )
    
    return {
        "parquet": report_parquet,
        "taxonomy": Path("test_data/joltax_cache"),
        "truth_dir": truth_dir
    }

def test_table_taxon_golden(golden_data, tmp_path):
    output_csv = tmp_path / "taxon.csv"
    generate_table_logic(
        input_parquet=golden_data["parquet"],
        output_file=output_csv,
        mode="taxon",
        taxonomy_dir=golden_data["taxonomy"],
        min_observed=0,
        min_reads=0
    )
    
    df_gen = pl.read_csv(output_csv).sort("t_id")
    df_truth = pl.read_csv(golden_data["truth_dir"] / "abundance_taxon.csv").sort("t_id")
    
    assert_frame_equal(df_gen, df_truth, check_column_order=False)

def test_table_clade_golden(golden_data, tmp_path):
    output_csv = tmp_path / "clade.csv"
    generate_table_logic(
        input_parquet=golden_data["parquet"],
        output_file=output_csv,
        mode="clade",
        taxonomy_dir=golden_data["taxonomy"],
        min_observed=0,
        min_reads=0
    )
    
    df_gen = pl.read_csv(output_csv).sort("t_id")
    df_truth = pl.read_csv(golden_data["truth_dir"] / "abundance_clade.csv").sort("t_id")
    
    assert_frame_equal(df_gen, df_truth, check_column_order=False)

def test_table_canonical_golden(golden_data, tmp_path):
    output_csv = tmp_path / "canonical.csv"
    generate_table_logic(
        input_parquet=golden_data["parquet"],
        output_file=output_csv,
        mode="canonical",
        taxonomy_dir=golden_data["taxonomy"],
        min_observed=0,
        min_reads=0
    )
    
    df_gen = pl.read_csv(output_csv).sort("t_id")
    df_truth = pl.read_csv(golden_data["truth_dir"] / "abundance_canonical.csv").sort("t_id")
    
    # In canonical mode, SweetBITS might include higher ranks that were empty in ground truth
    # Filter to only nodes present in truth
    df_gen = df_gen.filter(pl.col("t_id").is_in(df_truth["t_id"].to_list()))
    
    assert_frame_equal(df_gen, df_truth, check_column_order=False)

def test_table_filter_clade_bacteria(golden_data, tmp_path):
    output_csv = tmp_path / "bacteria.csv"
    generate_table_logic(
        input_parquet=golden_data["parquet"],
        output_file=output_csv,
        mode="clade",
        taxonomy_dir=golden_data["taxonomy"],
        clade_filter=2, # Bacteria
        min_observed=0,
        min_reads=0
    )
    
    df_gen = pl.read_csv(output_csv).sort("t_id")
    df_truth = pl.read_csv(golden_data["truth_dir"] / "abundance_clade_filtered_bacteria.csv").sort("t_id")
    
    assert_frame_equal(df_gen, df_truth, check_column_order=False)

def test_table_filter_min_reads(golden_data, tmp_path):
    output_csv = tmp_path / "min_reads.csv"
    generate_table_logic(
        input_parquet=golden_data["parquet"],
        output_file=output_csv,
        mode="taxon",
        taxonomy_dir=golden_data["taxonomy"],
        min_observed=0,
        min_reads=20
    )
    
    df_gen = pl.read_csv(output_csv).sort("t_id")
    df_truth = pl.read_csv(golden_data["truth_dir"] / "abundance_read_filtered_20.csv").sort("t_id")
    
    assert_frame_equal(df_gen, df_truth, check_column_order=False)

def test_table_filter_min_obs(golden_data, tmp_path):
    output_csv = tmp_path / "min_obs.csv"
    generate_table_logic(
        input_parquet=golden_data["parquet"],
        output_file=output_csv,
        mode="taxon",
        taxonomy_dir=golden_data["taxonomy"],
        min_observed=2,
        min_reads=0
    )
    
    df_gen = pl.read_csv(output_csv).sort("t_id")
    df_truth = pl.read_csv(golden_data["truth_dir"] / "abundance_obs_filtered_2.csv").sort("t_id")
    
    assert_frame_equal(df_gen, df_truth, check_column_order=False)
