import pytest
import polars as pl
from pathlib import Path
from sweetbits.annotate import annotate_table_logic
from sweetbits.metadata import get_standard_metadata, save_companion_metadata

@pytest.fixture
def mock_taxonomy(tmp_path):
    # This test will use the real test joltax cache if available,
    # or mock it if needed. Let's point to the test taxonomy.
    # The session context shows test_data/joltax_cache exists.
    return Path("test_data/joltax_cache")

@pytest.fixture
def base_table(tmp_path):
    df = pl.DataFrame({
        "t_id": [9606, 10090, 2],
        "sample_1": [10, 20, 30],
        "sample_2": [100, 200, 300]
    })
    path = tmp_path / "base.parquet"
    meta = get_standard_metadata("RAW_TABLE")
    df.write_parquet(path)
    save_companion_metadata(path, meta)
    return path

@pytest.fixture
def metadata_file(tmp_path):
    df = pl.DataFrame({
        "t_id": [9606, 10090],
        "status": ["Endangered", "Common"],
        "notes": ["Human", "Mouse"]
    })
    path = tmp_path / "metadata.csv"
    df.write_csv(path)
    return path

@pytest.fixture
def colliding_metadata(tmp_path):
    # Has a 'status' column that will collide
    df = pl.DataFrame({
        "t_id": [2, 9606],
        "status": ["Bacteria", "Mammal"],
        "other": ["X", "Y"]
    })
    path = tmp_path / "colliding_meta.csv"
    df.write_csv(path)
    return path

def test_annotate_table_basic(tmp_path, mock_taxonomy, base_table):
    out_path = tmp_path / "out.tsv"
    
    res = annotate_table_logic(
        input_table=base_table,
        taxonomy_dir=mock_taxonomy,
        output_file=out_path
    )
    
    assert res["taxa_processed"] == 3
    assert out_path.exists()
    
    df = pl.read_csv(out_path, separator="\t")
    
    # Check taxonomy columns are injected (t_scientific_name, etc.)
    assert "t_scientific_name" in df.columns
    assert "t_superkingdom" in df.columns or "t_domain" in df.columns
    
    # Check stats columns
    assert "mean_signal" in df.columns
    assert "median_signal" in df.columns
    
    # Check math for 9606 (sample_1=10, sample_2=100) -> mean=55, median=55
    row = df.filter(pl.col("t_id") == 9606)
    assert row["mean_signal"][0] == 55.0
    assert row["median_signal"][0] == 55.0
    
    # Check ordering (t_id first)
    assert df.columns[0] == "t_id"

def test_annotate_table_with_metadata(tmp_path, mock_taxonomy, base_table, metadata_file, colliding_metadata):
    out_path = tmp_path / "out.csv"
    
    res = annotate_table_logic(
        input_table=base_table,
        taxonomy_dir=mock_taxonomy,
        output_file=out_path,
        metadata_files=[metadata_file, colliding_metadata]
    )
    
    df = pl.read_csv(out_path)
    
    # Check metadata columns
    assert "status" in df.columns  # From first file
    assert "notes" in df.columns   # From first file
    assert "status_colliding_meta" in df.columns  # Renamed from second file
    assert "other" in df.columns   # From second file
    
    # Check left join preservation
    assert df.height == 3
    
    # 9606 is in both metadata files
    row_9606 = df.filter(pl.col("t_id") == 9606)
    assert row_9606["status"][0] == "Endangered"
    assert row_9606["status_colliding_meta"][0] == "Mammal"
    
    # 2 is only in colliding_meta
    row_2 = df.filter(pl.col("t_id") == 2)
    assert row_2["status"][0] is None
    assert row_2["status_colliding_meta"][0] == "Bacteria"

def test_missing_tid_column(tmp_path, mock_taxonomy, base_table):
    bad_meta = tmp_path / "bad.csv"
    pl.DataFrame({"id": [1, 2], "val": ["A", "B"]}).write_csv(bad_meta)
    
    out_path = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="must contain a 't_id' column"):
        annotate_table_logic(
            input_table=base_table,
            taxonomy_dir=mock_taxonomy,
            output_file=out_path,
            metadata_files=[bad_meta]
        )

def test_unsupported_metadata_format(tmp_path, mock_taxonomy, base_table):
    bad_ext = tmp_path / "meta.xlsx"
    bad_ext.touch() # Create dummy file
    
    out_path = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="Unsupported metadata file format"):
        annotate_table_logic(
            input_table=base_table,
            taxonomy_dir=mock_taxonomy,
            output_file=out_path,
            metadata_files=[bad_ext]
        )

def test_metadata_one_column_warning(tmp_path, mock_taxonomy, base_table, capsys):
    one_col = tmp_path / "one_col.csv"
    pl.DataFrame({"t_id": [9606, 10090]}).write_csv(one_col)
    
    out_path = tmp_path / "out.csv"
    annotate_table_logic(
        input_table=base_table,
        taxonomy_dir=mock_taxonomy,
        output_file=out_path,
        metadata_files=[one_col]
    )
    
    captured = capsys.readouterr()
    assert "only contains 1 column" in captured.err
