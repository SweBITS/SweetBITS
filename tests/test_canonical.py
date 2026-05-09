import pytest
import polars as pl
from pathlib import Path
from sweetbits.tables import generate_table_logic
from sweetbits.reports import gather_reports_logic
from polars.testing import assert_frame_equal

@pytest.fixture
def golden_data(tmp_path):
    """
    Sets up the Universal Golden Dataset for canonical tests.
    """
    base_dir = Path("test_data/universal_golden")
    input_dir = base_dir / "inputs"
    truth_dir = base_dir / "ground_truth"
    
    if not input_dir.exists():
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
        "taxonomy": Path("test_data/joltax_cache"),
        "truth_dir": truth_dir
    }

def test_canonical_mode_golden_dataset(golden_data, tmp_path):
    """
    Verifies that canonical mode correctly maps all reads to the standardized 
    NCBI ranks (Kingdom, Phylum, Class, Order, Family, Genus, Species) 
    using the awfully complex taxonomy.
    """
    output_csv = tmp_path / "canonical_table.csv"
    
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
    # to maintain tree structure. We filter to only nodes present in truth for parity.
    df_gen = df_gen.filter(pl.col("t_id").is_in(df_truth["t_id"].to_list()))
    
    assert_frame_equal(df_gen, df_truth, check_column_order=False)

def test_canonical_mode_mass_balance(golden_data, tmp_path):
    """
    Ensures that switching to canonical mode does not lose any reads.
    Total reads in canonical mode must exactly match total reads in taxon/clade mode.
    """
    output_canon = tmp_path / "canonical.parquet"
    output_taxon = tmp_path / "taxon.parquet"
    
    # 1. Run Canonical
    generate_table_logic(
        input_parquet=golden_data["parquet"],
        output_file=output_canon,
        mode="canonical",
        taxonomy_dir=golden_data["taxonomy"]
    )
    
    # 2. Run Taxon
    generate_table_logic(
        input_parquet=golden_data["parquet"],
        output_file=output_taxon,
        mode="taxon",
        taxonomy_dir=golden_data["taxonomy"]
    )
    
    df_canon = pl.read_parquet(output_canon)
    df_taxon = pl.read_parquet(output_taxon)
    
    # Sum all period columns
    period_cols = [c for r in [df_canon, df_taxon] for c in r.columns if c != "t_id"]
    period_cols = list(set(period_cols))
    
    canon_total = df_canon.select(pl.sum_horizontal(period_cols)).sum().item()
    taxon_total = df_taxon.select(pl.sum_horizontal(period_cols)).sum().item()
    
    assert canon_total == taxon_total
