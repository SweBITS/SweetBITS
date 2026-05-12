import pytest
import polars as pl
from pathlib import Path
from sweetbits.features import produce_feature_kmer_sample_logic, collect_feature_chunks_logic

def test_feature_chunks_equivalence(tmp_path):
    """
    Verifies that processing samples one-by-one and merging them (Divide and Conquer)
    produces the exact same result as processing all samples in one go.
    """
    input_dir = Path("test_data/universal_golden/inputs")
    taxonomy_dir = Path("test_data/joltax_cache")
    kmer_agg_dir = tmp_path / "kmer_agg"
    kmer_agg_dir.mkdir()
    
    # Pre-step: Generate kmer summary Parquets using existing tool
    from sweetbits.kmers import aggregate_kraken_kmers_logic
    for kraken_file in sorted(list(input_dir.glob("*.kraken")))[:3]:
        aggregate_kraken_kmers_logic(
            kraken_file=kraken_file,
            output_dir=kmer_agg_dir,
            taxonomy_dir=taxonomy_dir,
            overwrite=True
        )
        
    pattern = str(kmer_agg_dir / "*.kmers.parquet")
    
    # 1. Single-Pass Result
    single_pass_file = tmp_path / "single_pass.parquet"
    produce_feature_kmer_sample_logic(
        input_pattern=pattern,
        taxonomy_dir=taxonomy_dir,
        output_file=single_pass_file,
        overwrite=True
    )
    df_single = pl.read_parquet(single_pass_file).sort(["sample_id", "t_id"])
    
    # 2. Divide and Conquer Result
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    
    # Extract each sample individually
    for kmer_file in kmer_agg_dir.glob("*.kmers.parquet"):
        chunk_out = chunk_dir / f"{kmer_file.stem}.chunk.parquet"
        produce_feature_kmer_sample_logic(
            input_pattern=str(kmer_file),
            taxonomy_dir=taxonomy_dir,
            output_file=chunk_out,
            overwrite=True
        )
        
    # Merge chunks
    merged_file = tmp_path / "merged.parquet"
    collect_feature_chunks_logic(
        input_pattern=str(chunk_dir / "*.parquet"),
        output_file=merged_file,
        overwrite=True
    )
    df_merged = pl.read_parquet(merged_file).sort(["sample_id", "t_id"])
    
    # 3. Final Parity Check
    from polars.testing import assert_frame_equal
    assert_frame_equal(df_single, df_merged, check_column_order=False)
