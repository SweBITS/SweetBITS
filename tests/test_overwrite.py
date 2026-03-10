"""
tests.test_overwrite
Tests for overwrite protection logic across all SweetBITS tools.
"""

import pytest
import polars as pl
from pathlib import Path
from sweetbits.reports import gather_reports_logic
from sweetbits.tables import generate_table_logic
from sweetbits.reads import extract_reads_logic
from sweetbits.annotate import annotate_table_logic
from sweetbits.convert import convert_kraken_logic
from sweetbits.metadata import get_standard_metadata, save_companion_metadata

@pytest.fixture
def mock_report(tmp_path):
    report = tmp_path / "S1.report"
    # 8 columns
    report.write_text("0.1\t10\t10\t100\t50\tS\t9606\tHomo sapiens")
    return report

@pytest.fixture
def mock_report_parquet(tmp_path, mock_report):
    pfile = tmp_path / "merged.parquet"
    df = pl.DataFrame({
        "sample_id": ["S1"],
        "t_id": [9606],
        "taxon_reads": [10],
        "mm_tot": [100],
        "mm_uniq": [50],
        "source_file": ["S1.report"]
    })
    meta = get_standard_metadata("REPORT_PARQUET", source_path=tmp_path)
    df.write_parquet(pfile)
    save_companion_metadata(pfile, meta)
    return pfile

@pytest.fixture
def mock_kraken_parquet(tmp_path):
    pfile = tmp_path / "S1.kraken.parquet"
    df = pl.DataFrame({
        "sample_id": ["S1"],
        "read_id": ["R1"],
        "t_id": [9606],
        "mhg": [1],
        "r1_len": [150],
        "r2_len": [150],
        "total_len": [300],
        "kmer_string": ["9606:150"],
        "year": [2022],
        "week": [1],
        "r1_seq": ["A"],
        "r1_qual": ["I"],
        "r2_seq": ["T"],
        "r2_qual": ["J"]
    })
    meta = get_standard_metadata("KRAKEN_PARQUET")
    meta["has_fastq"] = "True"
    df.write_parquet(pfile)
    save_companion_metadata(pfile, meta)
    return pfile

def test_gather_reports_overwrite(tmp_path, mock_report):
    out = tmp_path / "out.parquet"
    out.touch()
    
    # Should fail
    with pytest.raises(FileExistsError, match="already exists"):
        gather_reports_logic(tmp_path, out, overwrite=False)
        
    # Should pass
    gather_reports_logic(tmp_path, out, overwrite=True)
    assert out.stat().st_size > 0

def test_table_overwrite(tmp_path, mock_report_parquet):
    out = tmp_path / "out.tsv"
    out.touch()
    
    tax_dir = Path("test_data/joltax_cache")
    
    # Should fail
    with pytest.raises(FileExistsError, match="already exists"):
        generate_table_logic(mock_report_parquet, out, taxonomy_dir=tax_dir, overwrite=False)
        
    # Should pass
    generate_table_logic(mock_report_parquet, out, taxonomy_dir=tax_dir, overwrite=True)
    assert out.stat().st_size > 0

def test_annotate_overwrite(tmp_path, mock_report_parquet):
    # Create a raw table
    raw = tmp_path / "raw.parquet"
    df = pl.DataFrame({"t_id": [9606], "S1": [10]})
    meta = get_standard_metadata("RAW_TABLE")
    df.write_parquet(raw)
    save_companion_metadata(raw, meta)
    
    out = tmp_path / "annotated.tsv"
    out.touch()
    
    # Mock taxonomy dir
    tax_dir = Path("test_data/joltax_cache")
    
    # Should fail
    with pytest.raises(FileExistsError, match="already exists"):
        annotate_table_logic(raw, tax_dir, out, overwrite=False)
        
    # Should pass
    annotate_table_logic(raw, tax_dir, out, overwrite=True)
    assert out.stat().st_size > 0

def test_convert_overwrite(tmp_path):
    k_file = tmp_path / "S1.kraken"
    k_file.write_text("U\tread1\t0\t150|150\t0\t0:116")
    
    out = tmp_path / "out.parquet"
    out.touch()
    
    # Should fail
    with pytest.raises(FileExistsError, match="already exists"):
        convert_kraken_logic(k_file, out, overwrite=False)
        
    # Should pass
    convert_kraken_logic(k_file, out, overwrite=True)
    assert out.stat().st_size > 0

def test_extract_reads_overwrite(tmp_path, mock_kraken_parquet):
    out_dir = tmp_path / "extract"
    out_dir.mkdir()
    
    tax_dir = Path("test_data/joltax_cache")
    
    # Try to get the actual name from the tree to be sure
    from joltax import JolTree
    tree = JolTree.load(str(tax_dir))
    name = tree.get_name(9606, strict=False) or "Unknown9606"
    from sweetbits.reads import format_short_name
    short_name = format_short_name(name)
    
    # Create an existing file that would be overwritten
    # Filename pattern: {sample_id}_{mode}_{tax_id}_{ShortName}_R[1/2].fastq.gz
    (out_dir / f"S1_clade_9606_{short_name}_R1.fastq.gz").touch()
    
    # Should fail
    with pytest.raises(FileExistsError, match="already exist"):
        extract_reads_logic(mock_kraken_parquet, tax_dir, [9606], out_dir, overwrite=False)
        
    # Should pass
    extract_reads_logic(mock_kraken_parquet, tax_dir, [9606], out_dir, overwrite=True)
