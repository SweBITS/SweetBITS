import pytest
import polars as pl
from pathlib import Path
from sweetbits.convert import convert_kraken_logic

@pytest.fixture
def mock_files(tmp_path):
    k_file = tmp_path / "Lj-2022_20_001.kraken"
    # Using 150|150 for length
    k_content = "U\tread1\t0\t150|150\t0\t0:116\nC\tread2\t562\t150|150\t2\t562:10 0:5 562:101\nC\tread3\t9606\t150|150\t1\t9606:116"
    k_file.write_text(k_content)

    r1_file = tmp_path / "S1_R1.fastq"
    r1_content = "@read1/1\nACGT\n+\nIIII\n@read2/1\nTGCA\n+\nJJJJ"
    r1_file.write_text(r1_content)
    
    r2_file = tmp_path / "S1_R2.fastq"
    r2_content = "@read1/2\nACGT\n+\nIIII\n@read2/2\nTGCA\n+\nJJJJ"
    r2_file.write_text(r2_content)
    
    return k_file, r1_file, r2_file

def test_convert_skinny(mock_files, tmp_path):
    k_file, _, _ = mock_files
    out_file = tmp_path / "out_skinny.parquet"
    
    res = convert_kraken_logic(k_file, out_file, no_fastq=True)
    assert res["records_processed"] == 3
    assert not res["has_fastq"]
    assert res["data_standard"] == "SWEBITS"
    
    df = pl.read_parquet(out_file)
    assert df.height == 3
    assert "r1_seq" not in df.columns
    # Check lengths parsed correctly
    assert df.filter(pl.col("read_id") == "read1")["r1_len"][0] == 150
    assert df.filter(pl.col("read_id") == "read1")["r2_len"][0] == 150
    assert df.filter(pl.col("read_id") == "read1")["total_len"][0] == 300
    
def test_convert_fat_with_depletion(mock_files, tmp_path):
    k_file, r1_file, r2_file = mock_files
    out_file = tmp_path / "out_fat.parquet"
    
    # Notice read3 is in Kraken but NOT in FASTQ (host-depletion simulation)
    res = convert_kraken_logic(k_file, out_file, r1_file=r1_file, r2_file=r2_file)
    
    assert res["records_processed"] == 3
    assert res["has_fastq"]
    
    df = pl.read_parquet(out_file)
    assert df.height == 3
    assert "r1_seq" in df.columns
    
    # Check read1
    r1 = df.filter(pl.col("read_id") == "read1")
    assert r1["r1_seq"][0] == "ACGT"
    
    # Check read3 (should be null for seq)
    r3 = df.filter(pl.col("read_id") == "read3")
    assert r3["r1_seq"][0] is None
    assert r3["r2_seq"][0] is None
    assert r3["t_id"][0] == 9606

def test_sync_error(mock_files, tmp_path):
    k_file, r1_file, r2_file = mock_files
    
    # Create out-of-sync FASTQ
    bad_r1 = tmp_path / "bad_R1.fastq"
    bad_r1.write_text("@read2/1\nTGCA\n+\nJJJJ\n@read1/1\nACGT\n+\nIIII")
    
    out_file = tmp_path / "out_bad.parquet"
    with pytest.raises(RuntimeError, match="out of sync"):
        convert_kraken_logic(k_file, out_file, r1_file=bad_r1, r2_file=r2_file)