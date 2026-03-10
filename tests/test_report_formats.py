import pytest
import polars as pl
from pathlib import Path
from sweetbits.testing import generate_mock_kraken_report_file
from sweetbits.reports import gather_reports_logic
from sweetbits.metadata import read_companion_metadata

def test_gather_reports_hyperloglog(tmp_path):
    report_dir = tmp_path / "reports_hll"
    report_dir.mkdir()
    
    generate_mock_kraken_report_file(report_dir / "S1.report", format="HYPERLOGLOG")
    
    output_parquet = tmp_path / "hll.parquet"
    gather_reports_logic(report_dir, output_parquet)
    
    df = pl.read_parquet(output_parquet)
    meta = read_companion_metadata(output_parquet)
    
    assert meta["report_format"] == "HYPERLOGLOG"
    assert "mm_tot" in df.columns
    assert "mm_uniq" in df.columns

def test_gather_reports_legacy(tmp_path):
    report_dir = tmp_path / "reports_legacy"
    report_dir.mkdir()
    
    generate_mock_kraken_report_file(report_dir / "S1.report", format="LEGACY")
    
    output_parquet = tmp_path / "legacy.parquet"
    gather_reports_logic(report_dir, output_parquet)
    
    df = pl.read_parquet(output_parquet)
    meta = read_companion_metadata(output_parquet)
    
    assert meta["report_format"] == "LEGACY"
    # mm columns should NOT be present in legacy
    assert "mm_tot" not in df.columns
    assert "mm_uniq" not in df.columns
    assert "taxon_reads" in df.columns

def test_gather_reports_mixed_consistency_error(tmp_path):
    report_dir = tmp_path / "reports_mixed"
    report_dir.mkdir()
    
    generate_mock_kraken_report_file(report_dir / "S1.report", format="HYPERLOGLOG")
    generate_mock_kraken_report_file(report_dir / "S2.report", format="LEGACY")
    
    output_parquet = tmp_path / "mixed.parquet"
    
    # Should raise ValueError because formats are mixed
    with pytest.raises(ValueError, match="Mixed report formats detected"):
        gather_reports_logic(report_dir, output_parquet)
