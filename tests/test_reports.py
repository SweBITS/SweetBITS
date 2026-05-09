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

def test_gather_reports_logic_swebits(tmp_path):
    # Setup: Create a directory with two mock reports (SweBITS style)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    
    report1 = report_dir / "Lj-2022_20_001.report"
    report2 = report_dir / "Ki-1974_02_001.report"
    
    generate_mock_kraken_report_file(report1)
    generate_mock_kraken_report_file(report2)
    
    output_parquet = tmp_path / "merged_swebits.parquet"
    
    gather_reports_logic(
        input_dir=report_dir,
        output_file=output_parquet
    )
    
    assert output_parquet.exists()
    df = pl.read_parquet(output_parquet)
    meta = read_companion_metadata(output_parquet)
    
    # Check Metadata
    assert meta["data_standard"] == "SWEBITS"
    
    # Check Columns
    assert "year" in df.columns
    assert "week" in df.columns
    
    # Check Sorting (1974 should be first)
    assert df["year"][0] == 1974

def test_gather_reports_logic_generic(tmp_path):
    # Setup: Reports with non-SweBITS names
    report_dir = tmp_path / "reports_generic"
    report_dir.mkdir()
    
    report1 = report_dir / "SampleA.report"
    report2 = report_dir / "SampleB.report"
    
    generate_mock_kraken_report_file(report1)
    generate_mock_kraken_report_file(report2)
    
    output_parquet = tmp_path / "merged_generic.parquet"
    
    gather_reports_logic(
        input_dir=report_dir,
        output_file=output_parquet
    )
    
    assert output_parquet.exists()
    df = pl.read_parquet(output_parquet)
    meta = read_companion_metadata(output_parquet)
    
    # Check Metadata
    assert meta["data_standard"] == "GENERIC"
    
    # Check Columns (should NOT have year/week)
    assert "year" not in df.columns
    assert "week" not in df.columns
    assert "sample_id" in df.columns
