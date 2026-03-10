"""
tests.test_canonical_complex
Stress tests for canonical remainder logic with complex taxonomic structures.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from sweetbits.tables import generate_table_logic
from sweetbits.metadata import save_companion_metadata, get_standard_metadata
from joltax import JolTree

@pytest.fixture
def complex_taxonomy(tmp_path):
    """Creates a deep and broad taxonomy with nested non-canonical ranks."""
    tax_dir = tmp_path / "complex_tax"
    tax_dir.mkdir()
    
    # Format: TID | ParentTID | Rank
    nodes = [
        "1\t|\t1\t|\tno rank\t|",
        "2\t|\t1\t|\tsuperkingdom\t|",
        "10\t|\t2\t|\tphylum\t|",
        "100\t|\t10\t|\tclass\t|",
        "1000\t|\t100\t|\tsubclass\t|",      # Non-canonical
        "10000\t|\t1000\t|\torder\t|",
        "100000\t|\t10000\t|\tfamily\t|",
        "1000000\t|\t100000\t|\tgenus\t|",    # Direct child of Family
        "10000000\t|\t1000000\t|\tspecies\t|",
        "10000001\t|\t10000000\t|\tsubspecies\t|", # Non-canonical child of Species
        
        "20\t|\t2\t|\tphylum\t|",
        "200\t|\t20\t|\tno rank\t|",           # Non-canonical parent
        "2000\t|\t200\t|\tgenus\t|",           # Child of non-canonical
        "3000\t|\t200\t|\tgenus\t|",           # Sibling Genus
    ]
    
    names = [f"{n.split()[0]}\t|\tName_{n.split()[0]}\t|\t\t|\tscientific name\t|" for n in nodes]
    
    with open(tax_dir / "nodes.dmp", "w") as f:
        for l in nodes: f.write(l + "\n")
    with open(tax_dir / "names.dmp", "w") as f:
        for l in names: f.write(l + "\n")
        
    tree = JolTree(tax_dir=str(tax_dir))
    cache_dir = tmp_path / "complex_cache"
    tree.save(str(cache_dir))
    return cache_dir

def test_canonical_deep_nesting(complex_taxonomy, tmp_path):
    """Tests remainder calculation through deep non-canonical gaps."""
    # Full exhaustive hierarchy from Root to Subspecies
    data = pl.DataFrame({
        "sample_id": ["S1"] * 10,
        "year": [2022] * 10, "week": [1] * 10,
        "t_id": [1, 2, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 10000001],
        "taxon_reads": [0, 50, 50, 50, 0, 50, 50, 50, 50, 50],
        "mm_tot": [0]*10, "mm_uniq": [0]*10, "source_file": ["f"]*10
    }).with_columns([pl.col("year").cast(pl.UInt16), pl.col("week").cast(pl.UInt8), pl.col("t_id").cast(pl.UInt32)])
    
    report_parquet = tmp_path / "deep.parquet"
    meta = get_standard_metadata("REPORT_PARQUET", source_path=tmp_path, data_standard="SWEBITS")
    data.write_parquet(report_parquet)
    save_companion_metadata(report_parquet, meta)
    
    out = tmp_path / "out.tsv"
    generate_table_logic(report_parquet, out, mode="canonical", taxonomy_dir=complex_taxonomy, min_observed=0, min_reads=0)
    
    res = {row["t_id"]: row["2022_01"] for row in pl.read_csv(out, separator="\t").to_dicts()}
    
    assert 10000001 not in res
    assert res[10000000] == 100 
    assert res[1000000] == 50   
    assert res[100000] == 50    
    assert res[10] == 50        

def test_canonical_broad_non_canonical_parent(complex_taxonomy, tmp_path):
    """Tests if multiple canonical children correctly subtract from a distant canonical ancestor."""
    # MUST be exhaustive to pass audit
    # Root(1) -> SK(2) -> Phylum(20) -> no-rank(200) -> [GenusB(2000), GenusC(3000)]
    data = pl.DataFrame({
        "sample_id": ["S1"] * 5,
        "year": [2022] * 5, "week": [1] * 5,
        "t_id": [1, 2, 20, 2000, 3000],
        "taxon_reads": [0, 0, 50, 100, 100],
        "mm_tot": [0]*5, "mm_uniq": [0]*5, "source_file": ["f"]*5
    }).with_columns([pl.col("year").cast(pl.UInt16), pl.col("week").cast(pl.UInt8), pl.col("t_id").cast(pl.UInt32)])
    
    report_parquet = tmp_path / "broad.parquet"
    meta = get_standard_metadata("REPORT_PARQUET", source_path=tmp_path, data_standard="SWEBITS")
    data.write_parquet(report_parquet)
    save_companion_metadata(report_parquet, meta)
    
    out = tmp_path / "out_broad.tsv"
    generate_table_logic(report_parquet, out, mode="canonical", taxonomy_dir=complex_taxonomy, min_observed=0, min_reads=0)
    
    res = {row["t_id"]: row["2022_01"] for row in pl.read_csv(out, separator="\t").to_dicts()}
    
    assert res[2000] == 100
    assert res[3000] == 100
    assert res[20] == 50

def test_canonical_adjacent_ranks(complex_taxonomy, tmp_path):
    """Tests behavior when canonical ranks are direct parents/children."""
    # Root(1) -> SK(2) -> Phylum(10) -> Class(100) -> Order(10000) -> Family(100000) -> Genus(1000000)
    data = pl.DataFrame({
        "sample_id": ["S1"] * 7,
        "year": [2022] * 7, "week": [1] * 7,
        "t_id": [1, 2, 10, 100, 10000, 100000, 1000000],
        "taxon_reads": [0, 0, 0, 0, 0, 0, 100],
        "mm_tot": [0]*7, "mm_uniq": [0]*7, "source_file": ["f"]*7
    }).with_columns([pl.col("year").cast(pl.UInt16), pl.col("week").cast(pl.UInt8), pl.col("t_id").cast(pl.UInt32)])
    
    report_parquet = tmp_path / "adjacent.parquet"
    meta = get_standard_metadata("REPORT_PARQUET", source_path=tmp_path, data_standard="SWEBITS")
    data.write_parquet(report_parquet)
    save_companion_metadata(report_parquet, meta)
    
    out = tmp_path / "out_adj.tsv"
    generate_table_logic(report_parquet, out, mode="canonical", taxonomy_dir=complex_taxonomy, min_observed=0, min_reads=0)
    
    res = {row["t_id"]: row["2022_01"] for row in pl.read_csv(out, separator="\t").to_dicts()}
    
    assert res[1000000] == 100
    assert res[100000] == 0
