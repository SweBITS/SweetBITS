import pytest
import polars as pl
import numpy as np
from pathlib import Path
from sweetbits.tables import generate_table_logic
from sweetbits.metadata import save_companion_metadata, get_standard_metadata
from joltax import JolTree

@pytest.fixture
def canonical_setup(tmp_path):
    # 1. Create a minimal taxonomy with non-canonical ranks
    taxonomy_dir = tmp_path / "taxonomy"
    taxonomy_dir.mkdir()
    
    # Root (1) -> Genus (5000000) -> Subgenus (555555) -> Species (5000001)
    nodes = [
        "1\t|\t1\t|\tno rank\t|",
        "5000000\t|\t1\t|\tgenus\t|",
        "555555\t|\t5000000\t|\tsubgenus\t|",
        "5000001\t|\t555555\t|\tspecies\t|",
    ]
    names = [
        "1\t|\troot\t|\t\t|\tscientific name\t|",
        "5000000\t|\tGenus_A\t|\t\t|\tscientific name\t|",
        "555555\t|\tSubgenus_X\t|\t\t|\tscientific name\t|",
        "5000001\t|\tSpecies_A\t|\t\t|\tscientific name\t|",
    ]
    
    with open(taxonomy_dir / "nodes.dmp", "w") as f:
        for line in nodes: f.write(line + "\n")
    with open(taxonomy_dir / "names.dmp", "w") as f:
        for line in names: f.write(line + "\n")
        
    tree = JolTree(tax_dir=str(taxonomy_dir))
    cache_dir = tmp_path / "joltax_cache"
    tree.save(str(cache_dir))
    
    # 2. Create custom report data
    # Root(1) [100] -> Genus(5000000) [35] -> Subgenus(555555) [25] -> Species(5000001) [20]
    report_data = pl.DataFrame({
        "sample_id": ["S1", "S1", "S1", "S1"],
        "year": [2022, 2022, 2022, 2022],
        "week": [1, 1, 1, 1],
        "t_id": [1, 5000000, 555555, 5000001],
        "taxon_reads": [65, 10, 5, 20],
        "mm_tot": [0, 0, 0, 0],
        "mm_uniq": [0, 0, 0, 0],
        "source_file": ["f1", "f1", "f1", "f1"]
    }).with_columns([
        pl.col("year").cast(pl.UInt16),
        pl.col("week").cast(pl.UInt8),
        pl.col("t_id").cast(pl.UInt32),
        pl.col("taxon_reads").cast(pl.UInt32)
    ])
    
    report_parquet = tmp_path / "report.parquet"
    meta = get_standard_metadata("REPORT_PARQUET", source_path=tmp_path, data_standard="SWEBITS")
    report_data.write_parquet(report_parquet)
    save_companion_metadata(report_parquet, meta)
    
    return {
        "parquet": report_parquet,
        "taxonomy": cache_dir
    }

def test_canonical_remainder_invalid_filter(canonical_setup, tmp_path):
    output_tsv = tmp_path / "fail.tsv"
    
    # 555555 is 'subgenus' in our setup, which is non-canonical.
    with pytest.raises(ValueError, match="is not a canonical rank"):
        generate_table_logic(
            input_parquet=canonical_setup["parquet"],
            output_file=output_tsv,
            mode="canonical",
            taxonomy_dir=canonical_setup["taxonomy"],
            clade_filter=555555
        )

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
    
    results = {row["t_id"]: row["2022_01"] for row in df.to_dicts()}
    
    assert 555555 not in results
    assert results[5000000] == 15 # (Clade 35) - (Child Species 20) = 15
    assert results[5000001] == 20
    assert results[1] == 65       # (Clade 100) - (Child Genus 35) = 65
