import pytest
import polars as pl
from pathlib import Path
from sweetbits.annotate import annotate_table_logic
from sweetbits.metadata import get_standard_metadata, save_companion_metadata

@pytest.fixture
def mock_taxonomy():
    return Path("test_data/joltax_cache")

def test_annotate_feature_table_string_crash(tmp_path, mock_taxonomy):
    # This test represents the reported crash
    df = pl.DataFrame({
        "t_id": [2, 9606],
        "feature_numeric": [10, 20],
        "feature_string": ["high", "low"]
    })
    path = tmp_path / "features.tsv"
    df.write_csv(path, separator="\t")
    
    out_path = tmp_path / "annotated.tsv"
    
    # Should not crash because stats are off by default
    res = annotate_table_logic(
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
    res = annotate_table_logic(
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
