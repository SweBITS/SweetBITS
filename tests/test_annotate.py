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
        output_file=out_path,
        add_stats=True
    )
    
    assert res["taxa_processed"] == 3
    assert out_path.exists()
    
    df = pl.read_csv(out_path, separator="\t")
    
    # Check taxonomy columns are injected (t_scientific_name, etc.)
    assert "t_scientific_name" in df.columns
    assert "t_superkingdom" in df.columns or "t_domain" in df.columns
    
    # Check stats columns
    assert "sig_avg" in df.columns
    assert "sig_med" in df.columns

    # Check math for 9606 (sample_1=10, sample_2=100) -> mean=55
    row = df.filter(pl.col("t_id") == 9606)
    assert row["sig_avg"][0] == 55.0    
    
    # Check full taxonomic accuracy (Homo sapiens)
    # The complex taxonomy provides these specific rank assignments
    assert row["t_superkingdom"][0] == "Eukaryota"
    assert row["t_phylum"][0] == "Chordata"
    assert row["t_class"][0] == "Mammalia"
    assert row["t_order"][0] == "Primates"
    assert row["t_genus"][0] == "Homo"
    assert row["t_scientific_name"][0] == "Homo sapiens"
    
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

def test_annotate_table_dfs_sorting(tmp_path, mock_taxonomy, base_table):
    out_path = tmp_path / "out_dfs.csv"
    
    # In base_table:
    # 2 (Bacteria): 165 mean
    # 10090 (Mouse): 110 mean
    # 9606 (Human): 55 mean
    
    annotate_table_logic(
        input_table=base_table,
        taxonomy_dir=mock_taxonomy,
        output_file=out_path,
        sort_order="dfs",
        add_stats=True
    )
    
    df = pl.read_csv(out_path)
    
    # 2 should be at the top if it's the heaviest branch
    assert df.height == 3
    assert "sig_avg" in df.columns
    
    # Verify we can at least run it
    tids = df["t_id"].to_list()
    assert 2 in tids
    assert 9606 in tids
    assert 10090 in tids

def test_annotate_table_dfs_sorting_weights(tmp_path, mock_taxonomy):
    # Setup a table where Eukaryota (9606, 10090) is heavier than Bacteria (2)
    df = pl.DataFrame({
        "t_id": [2, 9606, 10090],
        "sample_1": [10, 1000, 1], # Bacteria=10, Human=1000, Mouse=1
    })
    path = tmp_path / "weighted.parquet"
    meta = get_standard_metadata("RAW_TABLE")
    df.write_parquet(path)
    save_companion_metadata(path, meta)
    
    out_path = tmp_path / "out_dfs_weighted.csv"
    annotate_table_logic(
        input_table=path,
        taxonomy_dir=mock_taxonomy,
        output_file=out_path,
        sort_order="dfs",
        add_stats=True
    )
    
    df_res = pl.read_csv(out_path)
    tids = df_res["t_id"].to_list()
    # Eukaryota branch (9606, 10090) should come before Bacteria (2)
    # Human (9606) is much heavier than Mouse (1) so it comes first in its branch
    assert tids == [9606, 10090, 2]

def test_annotate_table_dfs_special_rows(tmp_path, mock_taxonomy):
    # Setup a table with special rows
    from sweetbits.utils import UNCLASSIFIED_TID, FILTERED_TID
    
    df = pl.DataFrame({
        "t_id": [2, UNCLASSIFIED_TID, FILTERED_TID, 9606],
        "sample_1": [10, 1000, 50, 1], 
    })
    path = tmp_path / "special.parquet"
    meta = get_standard_metadata("RAW_TABLE")
    df.write_parquet(path)
    save_companion_metadata(path, meta)
    
    out_path = tmp_path / "out_dfs_special.csv"
    annotate_table_logic(
        input_table=path,
        taxonomy_dir=mock_taxonomy,
        output_file=out_path,
        sort_order="dfs",
        add_stats=True
    )
    
    df_res = pl.read_csv(out_path)
    tids = df_res["t_id"].to_list()
    # Unclassified (0) then Filtered (10000000) then Tree (2, 9606)
    assert tids[0] == UNCLASSIFIED_TID
    assert tids[1] == FILTERED_TID
    assert tids[2:] == [2, 9606] # 2 (10) heavier than 9606 (1)

def test_annotate_table_alphabetical_special_rows(tmp_path, mock_taxonomy):
    # Setup a table with special rows
    from sweetbits.utils import UNCLASSIFIED_TID, FILTERED_TID
    
    df = pl.DataFrame({
        "t_id": [2, UNCLASSIFIED_TID, FILTERED_TID, 9606],
        "sample_1": [10, 1000, 50, 1], 
    })
    path = tmp_path / "special_alpha.parquet"
    meta = get_standard_metadata("RAW_TABLE")
    df.write_parquet(path)
    save_companion_metadata(path, meta)
    
    out_path = tmp_path / "out_alpha_special.csv"
    annotate_table_logic(
        input_table=path,
        taxonomy_dir=mock_taxonomy,
        output_file=out_path,
        sort_order="alphabetical"
    )
    
    df_res = pl.read_csv(out_path)
    tids = df_res["t_id"].to_list()
    # Unclassified (0) then Filtered (10000000) then Alphabetical Tree
    assert tids[0] == UNCLASSIFIED_TID
    assert tids[1] == FILTERED_TID
    # Alphabetical order: Bacteria (2) then Eukaryota (9606)
    assert tids[2:] == [2, 9606]

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

def test_annotate_feature_table_string_crash(tmp_path, mock_taxonomy):
    # This test represents a reported crash scenario
    df = pl.DataFrame({
        "t_id": [2, 9606],
        "feature_numeric": [10, 20],
        "feature_string": ["high", "low"]
    })
    path = tmp_path / "features.tsv"
    df.write_csv(path, separator="\t")
    
    out_path = tmp_path / "annotated.tsv"
    
    # Should not crash because stats are off by default
    annotate_table_logic(
        input_table=path,
        taxonomy_dir=mock_taxonomy,
        output_file=out_path,
        add_stats=False
    )
    
    assert out_path.exists()
    df_res = pl.read_csv(out_path, separator="\t")
    assert "t_scientific_name" in df_res.columns
    assert "sig_avg" not in df_res.columns
    assert "sig_med" not in df_res.columns

def test_annotate_with_stats_and_warnings(tmp_path, mock_taxonomy, capsys):
    df = pl.DataFrame({
        "t_id": [2, 9606],
        "sample_1": [10, 20],
        "sample_2": [100, 200],
        "meta_str": ["A", "B"]
    })
    path = tmp_path / "table.csv"
    df.write_csv(path)
    
    out_path = tmp_path / "annotated_stats.csv"
    
    # Test with stats enabled
    annotate_table_logic(
        input_table=path,
        taxonomy_dir=mock_taxonomy,
        output_file=out_path,
        add_stats=True
    )
    
    df_res = pl.read_csv(out_path)
    assert "sig_avg" in df_res.columns
    assert "sig_med" in df_res.columns
    
    # Check values
    # 9606: (20 + 200) / 2 = 110
    row_9606 = df_res.filter(pl.col("t_id") == 9606)
    assert row_9606["sig_avg"][0] == 110.0
    assert row_9606["sig_med"][0] == 110.0
    
    # Check warning for meta_str
    captured = capsys.readouterr()
    assert "Excluding non-numeric columns from summary statistics: ['meta_str']" in captured.err

def test_dfs_no_stats_warning(tmp_path, mock_taxonomy, capsys):
    df = pl.DataFrame({
        "t_id": [2, 9606],
        "val": [1, 2]
    })
    path = tmp_path / "dfs.csv"
    df.write_csv(path)
    
    out_path = tmp_path / "out_dfs.csv"
    
    # DFS without stats should warn
    annotate_table_logic(
        input_table=path,
        taxonomy_dir=mock_taxonomy,
        output_file=out_path,
        sort_order="dfs",
        add_stats=False
    )
    
    captured = capsys.readouterr()
    assert "DFS sorting requested without summary statistics" in captured.err
