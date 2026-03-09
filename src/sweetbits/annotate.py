"""
sweetbits.annotate
Logic for annotating abundance tables with taxonomy and external metadata.
"""

import polars as pl
import logging
import click
from pathlib import Path
from typing import Optional, List, Dict, Any
from joltax import JolTree
from sweetbits.metadata import validate_sweetbits_file, get_standard_metadata, save_companion_metadata
from sweetbits.utils import FILTERED_TID, UNCLASSIFIED_TID

logger = logging.getLogger(__name__)

def annotate_table_logic(
    input_table: Path,
    taxonomy_dir: Path,
    output_file: Path,
    metadata_files: Optional[List[Path]] = None,
    sort_order: str = "alphabetical",
    cores: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Annotates a raw abundance table with taxonomic lineages and external metadata.

    Loads a `<RAW_TABLE>`, queries JolTax for full taxonomic lineages (prefixed with 't_'),
    calculates summary statistics (mean/median), and joins any user-provided metadata files.
    Finally, sorts the table hierarchically and structures the columns.

    Args:
        input_table    : Path to the raw abundance table (Parquet, CSV, TSV).
        taxonomy_dir   : Path to the JolTax cache directory.
        output_file    : Path where the annotated table will be saved.
        metadata_files : Optional list of paths to external metadata files to join.
        sort_order     : Row sorting order ('alphabetical' or 'dfs').
        cores          : Number of CPU cores to use for Polars operations.
        overwrite      : Whether to overwrite the output file if it exists.

    Returns:
        A dictionary containing processing statistics:
        - 'taxa_processed' : Total number of taxa in the input table.
        - 'output_file'    : Path to the saved annotated table.

    Raises:
        ValueError      : If a metadata file lacks a 't_id' column or if an unsupported format is used.
        FileExistsError : If output_file exists and overwrite is False.
    """
    click.secho("Initiating table annotation...", fg="cyan", err=True)

    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --overwrite to replace it.")

    if metadata_files is None:
        metadata_files = []

    if cores:
        import os
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    # 1. Load Base Table
    ext = input_table.suffix.lower()

    # Track the standard if we have provenance, otherwise default to GENERIC
    data_standard = "GENERIC"
    mode = "taxon" # fallback if metadata missing

    if ext == ".parquet":
        metadata = validate_sweetbits_file(
            input_table, 
            expected_type="RAW_TABLE", 
            required_columns=["t_id"]
        )
        data_standard = metadata.get("data_standard", "GENERIC")
        mode = metadata.get("mode", "taxon")
        df = pl.read_parquet(input_table)
    else:
        # For CSV and TSV, try to validate the companion file if it exists
        try:
            metadata = validate_sweetbits_file(
                input_table, 
                expected_type="RAW_TABLE", 
                required_columns=["t_id"]
            )
            data_standard = metadata.get("data_standard", "GENERIC")
            mode = metadata.get("mode", "taxon")
        except FileNotFoundError:
            click.secho(
                f"Warning: Reading from {ext.upper()} without a companion metadata JSON file. "
                "Provenance metadata and version safety checks are bypassed.", 
                fg="yellow", err=True
            )

        if ext == ".tsv":
            df = pl.read_csv(input_table, separator="\t")
        else:
            df = pl.read_csv(input_table)

    if "t_id" not in df.columns:
        raise ValueError(f"Input table {input_table.name} must contain a 't_id' column.")

    sample_cols = [c for c in df.columns if c != "t_id"]
    base_tids = df["t_id"].to_list()

    # Extract special TaxIDs so JolTax doesn't process them
    has_filtered = FILTERED_TID in base_tids
    if has_filtered:
        base_tids = [tid for tid in base_tids if tid != FILTERED_TID]

    has_unclassified = UNCLASSIFIED_TID in base_tids
    if has_unclassified:
        base_tids = [tid for tid in base_tids if tid != UNCLASSIFIED_TID]

    base_tids_set = set(base_tids)
    num_taxa = len(base_tids)

    # 2. JolTax Annotation
    click.secho("Loading JolTax taxonomy tree...", fg="cyan", err=True)
    tree = JolTree.load(str(taxonomy_dir))

    # We enforce strict=True because the JolTax cache MUST match the Kraken database
    # used to generate the abundance table. Missing TaxIDs indicate a critical config error.
    tax_df = tree.annotate(base_tids, strict=True)
    
    # Ensure t_id is UInt32 to accommodate large IDs like FILTERED_TID
    tax_df = tax_df.with_columns(pl.col("t_id").cast(pl.UInt32))
    
    click.secho(f"Annotated {num_taxa}/{num_taxa} taxa using JolTax taxonomy", fg="green", err=True)

    # Re-inject special synthetic rows
    special_rows = []
    if has_unclassified:
        unclass_row = {"t_id": [UNCLASSIFIED_TID], "t_scientific_name": ["unclassified"], "t_rank": ["no rank"]}
        special_rows.append(unclass_row)

    if has_filtered:
        filtered_row = {"t_id": [FILTERED_TID], "t_scientific_name": ["Filtered classified"], "t_rank": ["synthetic"]}
        special_rows.append(filtered_row)

    if special_rows:
        for row in special_rows:
            for col in tax_df.columns:
                if col not in row:
                    row[col] = [None]

            row_df = pl.DataFrame(row, schema=tax_df.schema)
            tax_df = pl.concat([tax_df, row_df])

    tax_cols = tax_df.columns
    df = df.join(tax_df, on="t_id", how="left")

    # 3. Metadata Loop
    metadata_cols = []
    for m_path in metadata_files:
        m_ext = m_path.suffix.lower()
        if m_ext == ".parquet":
            m_df = pl.read_parquet(m_path)
        elif m_ext == ".tsv":
            m_df = pl.read_csv(m_path, separator="\t")
        elif m_ext == ".csv":
            m_df = pl.read_csv(m_path)
        else:
            raise ValueError(f"Unsupported metadata file format '{m_ext}' for {m_path.name}. Supported formats are .csv, .tsv, .parquet")

        if "t_id" not in m_df.columns:
            raise ValueError(f"Metadata file {m_path.name} must contain a 't_id' column.")

        if len(m_df.columns) == 1:
            click.secho(
                f"Warning: Metadata file {m_path.name} only contains 1 column ('{m_df.columns[0]}'). "
                f"This might indicate a separator mismatch or an empty metadata file.", 
                fg="yellow", err=True
            )

        m_tids = set(m_df["t_id"].to_list())
        intersect = len(base_tids_set.intersection(m_tids))
        click.secho(f"Annotated {intersect}/{num_taxa} taxa using {m_path.name}", fg="green", err=True)

        current_cols = set(df.columns)
        rename_map = {}
        new_m_cols = []

        for c in m_df.columns:
            if c == "t_id":
                continue
            if c in current_cols:
                new_name = f"{c}_{m_path.stem}"
                rename_map[c] = new_name
                new_m_cols.append(new_name)
                click.secho(
                    f"Warning: Column collision for '{c}' from {m_path.name}. Renamed to '{new_name}'.", 
                    fg="yellow", err=True
                )
            else:
                new_m_cols.append(c)

        if rename_map:
            m_df = m_df.rename(rename_map)

        df = df.join(m_df, on="t_id", how="left")
        metadata_cols.extend(new_m_cols)

    # 4. Summary Statistics
    click.secho("Calculating summary statistics...", fg="cyan", err=True)
    if sample_cols:
        df = df.with_columns([
            pl.mean_horizontal(sample_cols).alias("mean_signal")
        ])
    else:
        df = df.with_columns([
            pl.lit(0.0).alias("mean_signal")
        ])

    # 5. Sorting
    if sort_order == "dfs":
        click.secho("Applying abundance-weighted DFS sorting...", fg="cyan", err=True)
        import numpy as np

        # 5.1 Calculate Clade Weights
        # Map TaxIDs to their mean signals
        table_tid_means = dict(zip(df["t_id"].to_list(), df["mean_signal"].to_list()))

        num_nodes = len(tree.parents)
        clade_weights = np.zeros(num_nodes, dtype=np.float64)

        table_indices = tree._get_indices(np.array(list(table_tid_means.keys())))
        for i, idx in enumerate(table_indices):
            if idx != -1:
                clade_weights[idx] = table_tid_means[list(table_tid_means.keys())[i]]

        # Propagate weights up the tree
        indices_by_depth = np.argsort(tree.depths)[::-1]
        for idx in indices_by_depth:
            p_idx = tree.parents[idx]
            if p_idx != idx and p_idx != -1:
                if mode == "clade":
                    clade_weights[p_idx] = max(clade_weights[p_idx], clade_weights[idx])
                else:
                    clade_weights[p_idx] += clade_weights[idx]

        # 5.2 Build Children List
        children = [[] for _ in range(num_nodes)]
        for idx, p_idx in enumerate(tree.parents):
            if idx != p_idx and p_idx != -1:
                children[p_idx].append(idx)

        # 5.3 Sort siblings by clade weight (descending)
        for i in range(num_nodes):
            if children[i]:
                children[i].sort(key=lambda x: clade_weights[x], reverse=True)

        # 5.4 Execute DFS Traversal
        dfs_order = []

        # Prepend Unclassified and Filtered reads if present
        if UNCLASSIFIED_TID in table_tid_means:
            dfs_order.append(UNCLASSIFIED_TID)
        if has_filtered:
            dfs_order.append(FILTERED_TID)

        stack = [0] # JolTree root index is 0
        visited_tids = set(dfs_order)

        while stack:
            idx = stack.pop()
            tid = int(tree._index_to_id[idx])

            if tid in table_tid_means and tid not in visited_tids:
                dfs_order.append(tid)
                visited_tids.add(tid)

            # Reverse order for stack to process heaviest first
            for c_idx in reversed(children[idx]):
                stack.append(c_idx)

        # Create a mapping for Polars sort
        order_df = pl.DataFrame({
            "t_id": dfs_order,
            "sort_index": range(len(dfs_order))
        }).with_columns(pl.col("t_id").cast(pl.UInt32))

        df = df.join(order_df, on="t_id", how="left").sort("sort_index").drop("sort_index")
    else: # Hierarchical Alphabetical
        click.secho("Applying hierarchical taxonomic sort...", fg="cyan", err=True)
        sort_cols_target = [
            "t_domain", "t_superkingdom", "t_phylum", "t_class", "t_order", 
            "t_family", "t_genus", "t_species", "t_id"
        ]
        actual_sort_cols = [c for c in sort_cols_target if c in df.columns]

        # Use case-insensitive sorting for string columns to avoid ASCII E < d issues
        # and ensure Unclassified/Filtered are always at the top
        sort_exprs = [
            # Priority 1: Unclassified (0)
            (pl.col("t_id") != UNCLASSIFIED_TID).cast(pl.UInt8),
            # Priority 2: Filtered Classified (MaxInt)
            (pl.col("t_id") != FILTERED_TID).cast(pl.UInt8),
        ]

        for c in actual_sort_cols:
            if c == "t_id":
                sort_exprs.append(pl.col(c))
            else:
                # We lowercase the string for sorting purposes only
                sort_exprs.append(pl.col(c).cast(pl.Utf8).str.to_lowercase())

        df = df.sort(sort_exprs, nulls_last=True)
    # 6. Column Ordering
    ordered_cols = tax_cols + metadata_cols + ["mean_signal"] + sample_cols
    # Ensure t_id is first if it isn't already (though JolTax annotate puts it first)
    if ordered_cols[0] != "t_id":
        ordered_cols.remove("t_id")
        ordered_cols.insert(0, "t_id")

    df = df.select(ordered_cols)

    # 7. Output Generation
    click.secho(f"Saving annotated table to {output_file.name}...", fg="cyan", err=True)
    out_ext = output_file.suffix.lower()
    meta = get_standard_metadata(
        "ANNOTATED_TABLE", 
        source_path=input_table, 
        sorting="Taxonomic Hierarchy" if sort_order == "alphabetical" else "Abundance-Weighted DFS", 
        data_standard=data_standard
    )

    if out_ext == ".parquet":
        df.write_parquet(output_file)
    elif out_ext == ".tsv":
        df.write_csv(output_file, separator="\t")
    else:
        df.write_csv(output_file)

    save_companion_metadata(output_file, meta)

    click.secho("Done!", fg="cyan", bold=True, err=True)

    return {
        "taxa_processed": num_taxa,
        "output_file": str(output_file)
    }
