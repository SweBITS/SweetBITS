"""
sweetbits.kmers
Logic for aggregating k-mer hits and read length distributions from Kraken 2 read-by-read data.
"""

import os
import click
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from joltax import JolTree
from sweetbits.utils import get_sample_info, check_write_permission
from sweetbits.metadata import get_standard_metadata, save_companion_metadata

def aggregate_kraken_kmers_logic(
    kraken_file: Path,
    output_dir: Path,
    taxonomy_dir: Path,
    cores: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Processes a Kraken 2 read-by-read file and aggregates k-mer counts and 
    read lengths for species-level clades.

    Args:
        kraken_file  : Path to the Kraken 2 read-by-read output file.
        output_dir   : Directory to save the resulting Parquet files.
        taxonomy_dir : Path to the JolTax cache directory.
        cores        : Number of threads for Polars to use.
        overwrite    : Whether to overwrite existing output files.

    Returns:
        A dictionary with processing statistics.
    """
    if cores:
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    # 1. Identify Sample & Setup Paths
    info = get_sample_info(kraken_file.name)
    sample_id = info["sample_id"]
    data_standard = info["data_standard"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    kmer_out = output_dir / f"{sample_id}.kmers.parquet"
    len_out = output_dir / f"{sample_id}.read_lengths.parquet"

    if (kmer_out.exists() or len_out.exists()) and not overwrite:
        raise FileExistsError(f"Output files for {sample_id} already exist. Use --overwrite to replace.")

    check_write_permission(output_dir)

    # 2. Load Taxonomy
    click.secho(f"Loading JolTax taxonomy from {taxonomy_dir.name}...", fg="cyan", err=True)
    tree = JolTree.load(taxonomy_dir)

    # 3. Pre-scan: Identify relevant species clades
    click.secho("Phase 1/3: Scanning TaxIDs to identify species clades...", fg="cyan", err=True)
    
    # Fast scan to find unique TaxIDs that are classified
    present_tids_df = (
        pl.scan_csv(
            kraken_file,
            separator='\t',
            has_header=False,
            new_columns=["status", "read_id", "t_id_raw", "length", "mhg", "kmer_str"],
            schema_overrides={"t_id_raw": pl.UInt32, "length": pl.String}
        )
        .filter(pl.col("status") == "C")
        .select("t_id_raw")
        .unique()
        .collect()
    )
    present_tids = present_tids_df["t_id_raw"].to_list()
    
    if not present_tids:
        click.secho("Warning: No classified reads found in the input file.", fg="yellow", err=True)
        return {"status": "Skipped (No classified reads)", "sample_id": sample_id}

    # Identify species-level clades for all present TaxIDs using vectorized walk-up
    indices = tree._get_indices(np.array(present_tids, dtype=np.uint32))
    valid_mask = indices != -1
    valid_indices = indices[valid_mask]
    valid_tids = np.array(present_tids, dtype=np.uint32)[valid_mask]

    species_idx = -1
    try:
        species_idx = tree.rank_names.index("species")
    except ValueError:
        click.secho("Warning: 'species' rank not found in taxonomy tree.", fg="yellow", err=True)

    tid_to_species_root = {}
    if species_idx != -1:
        current_indices = valid_indices.copy()
        species_ancestor_indices = np.full_like(valid_indices, -1)
        
        # Walk up the tree layer by layer to find the first 'species' ancestor
        # We cap the iterations at the maximum depth of the tree
        max_depth = int(tree.depths.max()) + 1 if len(tree.depths) > 0 else 1
        for _ in range(max_depth):
            is_species = (tree.ranks[current_indices] == species_idx)
            # Record ancestor if it's a species and we haven't found one for this lineage yet
            update_mask = (species_ancestor_indices == -1) & is_species
            species_ancestor_indices[update_mask] = current_indices[update_mask]
            
            # Move to parents
            next_indices = tree.parents[current_indices]
            # Stop if all nodes hit root or no more changes possible
            if np.all(next_indices == current_indices):
                break
            current_indices = next_indices

        # Build mapping dictionary for use in Polars
        for i, tid in enumerate(valid_tids):
            anc_idx = species_ancestor_indices[i]
            if anc_idx != -1:
                tid_to_species_root[int(tid)] = int(tree._index_to_id[anc_idx])

    if not tid_to_species_root:
        click.secho("Warning: No reads were classified to species-level clades.", fg="yellow", err=True)
        return {"status": "Skipped (No species clades)", "sample_id": sample_id}

    # Create a mapping DataFrame for Polars join
    clade_map_df = pl.DataFrame({
        "t_id_raw": list(tid_to_species_root.keys()),
        "t_id": list(tid_to_species_root.values())
    }).with_columns(pl.col("t_id_raw").cast(pl.UInt32), pl.col("t_id").cast(pl.UInt32))

    # 4. Phase 2: Streaming Aggregation
    click.secho("Phase 2/3: Aggregating k-mer hits and read lengths...", fg="cyan", err=True)
    
    # Base LazyFrame
    # Kraken Read-by-Read Format: Status, ReadID, TaxID, Length (R1|R2), MHG, KmerString
    base_lf = (
        pl.scan_csv(
            kraken_file,
            separator='\t',
            has_header=False,
            new_columns=["status", "read_id", "t_id_raw", "length", "mhg", "kmer_str"],
            schema_overrides={"t_id_raw": pl.UInt32, "length": pl.String}
        )
        .filter(pl.col("status") == "C")
        # Inner join with clade_map_df effectively filters for only species-clade reads
        .join(clade_map_df.lazy(), on="t_id_raw", how="inner")
    )

    # --- K-mer Aggregation Branch ---
    # Parse the k-mer string: "taxid:count taxid:count ..."
    kmer_agg_lf = (
        base_lf
        .select(["t_id", "kmer_str"])
        # Extract all "taxid:count" pairs
        .with_columns(pl.col("kmer_str").str.extract_all(r"(\d+:\d+)").alias("pairs"))
        .explode("pairs")
        # Split "taxid:count" into two columns
        .with_columns(
            pl.col("pairs").str.split_exact(":", 1).alias("pairs")
        )
        .unnest("pairs")
        .rename({"field_0": "k_tid", "field_1": "k_count"})
        .with_columns([
            pl.col("k_tid").cast(pl.UInt32),
            # Cast to UInt64 to prevent integer overflow during species-level aggregation
            pl.col("k_count").cast(pl.UInt64)
        ])
        # Aggregate
        .group_by(["t_id", "k_tid"])
        .agg(pl.sum("k_count").alias("kmer_count"))
        .rename({"k_tid": "kmer_tax_id"})
        .sort(["t_id", "kmer_tax_id"])
    )

    # --- Read Length Aggregation Branch ---
    # Parse length: "R1_len|R2_len"
    len_agg_lf = (
        base_lf
        .select(["t_id", "length"])
        .with_columns(
            pl.col("length").str.split_exact("|", 1).alias("length")
        )
        .unnest("length")
        .with_columns(
            (pl.col("field_0").cast(pl.UInt16) + pl.col("field_1").cast(pl.UInt16).fill_null(0)).alias("read_length")
        )
        .group_by(["t_id", "read_length"])
        .agg(pl.len().alias("read_count"))
        .sort(["t_id", "read_length"])
    )

    # 5. Phase 3: Sink & Metadata
    display_dir = "the current directory" if str(output_dir) == "." else f"'{output_dir.name}'"
    click.secho(f"Phase 3/3: Writing results to {display_dir}...", fg="cyan", err=True)

    # Inject sample metadata into results
    def inject_meta(lf: pl.LazyFrame) -> pl.LazyFrame:
        cols = {"sample_id": pl.lit(sample_id).cast(pl.Categorical)}
        if data_standard == "SWEBITS":
            cols["year"] = pl.lit(info["year"]).cast(pl.UInt16)
            cols["week"] = pl.lit(info["week"]).cast(pl.UInt8)
        return lf.with_columns(**cols)

    kmer_agg_df = inject_meta(kmer_agg_lf).collect(engine="streaming")
    len_agg_df = inject_meta(len_agg_lf).collect(engine="streaming")

    # Save K-mers
    kmer_agg_df.write_parquet(kmer_out, compression="zstd")
    kmer_meta = get_standard_metadata(
        file_type="KMER_AGG_PARQUET",
        source_path=kraken_file,
        compression="zstd",
        sorting="t_id, kmer_tax_id",
        data_standard=data_standard
    )
    save_companion_metadata(kmer_out, kmer_meta)

    # Save Read Lengths
    len_agg_df.write_parquet(len_out, compression="zstd")
    len_meta = get_standard_metadata(
        file_type="READ_LEN_PARQUET",
        source_path=kraken_file,
        compression="zstd",
        sorting="t_id, read_length",
        data_standard=data_standard
    )
    save_companion_metadata(len_out, len_meta)

    return {
        "sample_id": sample_id,
        "species_clades_found": len(set(tid_to_species_root.values())),
        "total_kmer_records": kmer_agg_df.height,
        "total_read_len_records": len_agg_df.height,
        "status": "Success"
    }
