import pytest
import polars as pl
import numpy as np
from pathlib import Path
from sweetbits.tables import generate_table_logic
from sweetbits.metadata import write_parquet_with_metadata, get_standard_metadata
from joltax import JolTree

@pytest.fixture
def canonical_setup(tmp_path):
    # 1. Create a custom taxonomy
    taxonomy_dir = tmp_path / "taxonomy"
    taxonomy_dir.mkdir()
    
    # 1 (Root, Canonical)
    # 2 (Bacteria, Canonical)
    # 5000000 (MockGenus, Canonical)
    # 555555 (MockNonCanonical, rank subgenus, child of 5000000)
    # 5000001 (MockSpecies, Canonical, child of 555555)
    
    nodes = [
        "1\t|\t1\t|\tno rank\t|",
        "2\t|\t1\t|\tsuperkingdom\t|",
        "5000000\t|\t2\t|\tgenus\t|",
        "555555\t|\t5000000\t|\tsubgenus\t|", # Non-canonical rank
        "5000001\t|\t555555\t|\tspecies\t|",
    ]
    
    names = [
        "1\t|\troot\t|\t\t|\tscientific name\t|",
        "2\t|\tBacteria\t|\t\t|\tscientific name\t|",
        "5000000\t|\tMockGenus\t|\t\t|\tscientific name\t|",
        "555555\t|\tMockNonCanonical\t|\t\t|\tscientific name\t|",
        "5000001\t|\tMockSpecies\t|\t\t|\tscientific name\t|",
    ]
    
    with open(taxonomy_dir / "nodes.dmp", "w") as f:
        for line in nodes: f.write(line + "\n")
    with open(taxonomy_dir / "names.dmp", "w") as f:
        for line in names: f.write(line + "\n")
        
    tree = JolTree(tax_dir=str(taxonomy_dir))
    cache_dir = tmp_path / "joltax_cache"
    tree.save(str(cache_dir))
    
    # 2. Create custom report data (REPORT_PARQUET format)
    # Sample S1 (2022_01)
    # Total at Genus (5000000) is 35 (includes child 555555 and grandchild 5000001)
    # 555555 has 25 clade reads.
    # 5000001 has 20 clade reads.
    
    report_data = pl.DataFrame({
        "sample_id": ["S1", "S1", "S1"],
        "year": [2022, 2022, 2022],
        "week": [1, 1, 1],
        "t_id": [5000000, 555555, 5000001],
        "clade_reads": [35, 25, 20],
        "taxon_reads": [10, 5, 20], # Remainder at 5000000 should be 35 - 20 = 15
        "mm_tot": [0, 0, 0],
        "mm_uniq": [0, 0, 0],
        "source_file": ["f1", "f1", "f1"]
    }).with_columns([
        pl.col("year").cast(pl.UInt16),
        pl.col("week").cast(pl.UInt8),
        pl.col("t_id").cast(pl.UInt32),
        pl.col("clade_reads").cast(pl.UInt32),
        pl.col("taxon_reads").cast(pl.UInt32),
    ])
    
    report_parquet = tmp_path / "report.parquet"
    meta = get_standard_metadata("REPORT_PARQUET", source_path=tmp_path)
    write_parquet_with_metadata(report_data, report_parquet, meta)
    
    return {
        "parquet": report_parquet,
        "taxonomy": cache_dir
    }

def test_canonical_remainder_logic(canonical_setup, tmp_path):
    output_tsv = tmp_path / "canonical_table.tsv"
    
    generate_table_logic(
        input_parquet=canonical_setup["parquet"],
        output_file=output_tsv,
        mode="canonical",
        taxonomy_dir=canonical_setup["taxonomy"],
        min_observed=1,
        min_reads=1
    )
    
    df = pl.read_csv(output_tsv, separator="\t")
    print("\nDEBUG: Result DataFrame:")
    print(df)
    
    # Expected results:
    # 5000001 (Species) voted 20 into 5000000 (Genus, its NCA).
    # 555555 (Subgenus) is IGNORED as a canonical target.
    # Remainder(5000000) = Clade(5000000) - Clade(5000001) = 35 - 20 = 15.
    # Remainder(5000001) = Clade(5000001) - 0 = 20.
    
    results = {row["t_id"]: row["2022_01"] for row in df.to_dicts()}
    
    assert 555555 not in results
    assert results[5000000] == 15
    assert results[5000001] == 20
