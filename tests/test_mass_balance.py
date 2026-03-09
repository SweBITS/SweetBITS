"""
tests.test_mass_balance
Tests for the mass balance audit in canonical remainder calculation.
"""

import pytest
import polars as pl
from pathlib import Path
from sweetbits.tables import generate_table_logic
from sweetbits.metadata import save_companion_metadata, get_standard_metadata
from joltax import JolTree

@pytest.fixture
def simple_tax(tmp_path):
    tax_dir = tmp_path / "tax"
    tax_dir.mkdir()
    nodes = [
        "1\t|\t1\t|\tno rank\t|",
        "10\t|\t1\t|\tgenus\t|",
        "100\t|\t10\t|\tspecies\t|",
    ]
    names = [f"{n.split()[0]}\t|\tName\t|\t\t|\tscientific name\t|" for n in nodes]
    with open(tax_dir / "nodes.dmp", "w") as f:
        for l in nodes: f.write(l + "\n")
    with open(tax_dir / "names.dmp", "w") as f:
        for l in names: f.write(l + "\n")
    tree = JolTree(tax_dir=str(tax_dir))
    cache_dir = tmp_path / "cache"
    tree.save(str(cache_dir))
    return cache_dir

def test_mass_balance_failure(simple_tax, tmp_path):
    """Tests that an inconsistent report triggers a RuntimeError."""
    # Parent (10) has 100 reads.
    # Child (100) has 120 reads (IMPOSSIBLE).
    data = pl.DataFrame({
        "sample_id": ["S1", "S1"],
        "year": [2022, 2022], "week": [1, 1],
        "t_id": [10, 100],
        "clade_reads": [100, 120], # This breaks mass balance
        "taxon_reads": [0, 120],
        "mm_tot": [0, 0], "mm_uniq": [0, 0], "source_file": ["f", "f"]
    }).with_columns([pl.col("year").cast(pl.UInt16), pl.col("week").cast(pl.UInt8), pl.col("t_id").cast(pl.UInt32)])
    
    report_parquet = tmp_path / "broken.parquet"
    meta = get_standard_metadata("REPORT_PARQUET", source_path=tmp_path, data_standard="SWEBITS")
    data.write_parquet(report_parquet)
    save_companion_metadata(report_parquet, meta)
    
    out = tmp_path / "out.tsv"
    
    # This should raise RuntimeError during audit
    with pytest.raises(RuntimeError, match="Mass balance check failed"):
        generate_table_logic(report_parquet, out, mode="canonical", taxonomy_dir=simple_tax, min_observed=0, min_reads=0)

def test_mass_balance_success(simple_tax, tmp_path):
    """Tests that a consistent report passes the audit."""
    data = pl.DataFrame({
        "sample_id": ["S1", "S1", "S1"],
        "year": [2022, 2022, 2022], "week": [1, 1, 1],
        "t_id": [1, 10, 100],
        "clade_reads": [100, 80, 50], # Consistent
        "taxon_reads": [20, 30, 50],
        "mm_tot": [0, 0, 0], "mm_uniq": [0, 0, 0], "source_file": ["f", "f", "f"]
    }).with_columns([pl.col("year").cast(pl.UInt16), pl.col("week").cast(pl.UInt8), pl.col("t_id").cast(pl.UInt32)])
    
    report_parquet = tmp_path / "perfect.parquet"
    meta = get_standard_metadata("REPORT_PARQUET", source_path=tmp_path, data_standard="SWEBITS")
    data.write_parquet(report_parquet)
    save_companion_metadata(report_parquet, meta)
    
    out = tmp_path / "out_ok.tsv"
    # This should pass without error
    generate_table_logic(report_parquet, out, mode="canonical", taxonomy_dir=simple_tax, min_observed=0, min_reads=0)
    assert out.exists()
