import click
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from sweetbits import __version__
from sweetbits.reports import gather_reports_logic
from sweetbits.tables import generate_table_logic
from sweetbits.metadata import read_parquet_metadata
from sweetbits.reads import extract_reads_logic
from sweetbits.annotate import annotate_table_logic
from sweetbits.convert import convert_kraken_logic

def print_header(ctx):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    click.echo(f"{'Start time':20}: {now}", err=True)
    click.echo(f"{'Toolkit Version':20}: {__version__}", err=True)
    click.echo(f"{'CWD':20}: {os.getcwd()}", err=True)
    click.echo(f"{'Command':20}: sweetbits {' '.join(sys.argv[1:])}", err=True)
    click.echo("-" * 60, err=True)

def print_parameters(params):
    click.secho("Parameters:", fg="bright_black", bold=True, err=True)
    for k, v in params.items():
        label = k.replace("_", " ").title()
        click.echo(click.style(f"  {label:18}: ", fg="bright_black") + click.style(f"{v}", fg="bright_black"), err=True)
    click.echo("-" * 60, err=True)

def print_footer(start_time, summary_dict=None):
    elapsed = time.time() - start_time
    click.echo("-" * 60, err=True)
    if summary_dict:
        for k, v in summary_dict.items():
            label = k.replace("_", " ").title()
            click.echo(f"{label:20}: {v}", err=True)
            
    click.echo(f"{'Time elapsed':20}: {elapsed:.2f}s", err=True)
    click.echo(f"{'End time':20}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", err=True)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    """
    SweetBITS: Bioinformatics command-line tools for the Swedish Biodiversity 
    in Time and Space (SweBITS) project.
    """
    pass

@main.command(short_help="Merge Kraken reports into a single Parquet file.")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Path to output Parquet file.")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories (Default: True).")
@click.option("--include", "-i", default="*.report", help="Pattern to match report files (Default: *.report).")
def gather_reports(directory, output, recursive, include):
    """
    Finds and merges multiple 8-column Kraken reports into a single Polars-optimized 
    Parquet file, including provenance and temporal metadata.
    """
    start_time = time.time()
    ctx = click.get_current_context()
    print_header(ctx)
    print_parameters(ctx.params)
    
    try:
        summary = gather_reports_logic(
            input_dir=directory,
            output_file=output,
            recursive=recursive,
            include_pattern=include
        )
        summary["status"] = "Success"
        summary["output_file"] = str(output)
        print_footer(start_time, summary)
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)

@main.command(short_help="Generate abundance tables from merged reports.")
@click.argument("input_parquet", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Path to output file (.csv, .tsv, .parquet).")
@click.option("--mode", "-m", type=click.Choice(["taxon", "clade", "canonical"]), default="clade", help="Abundance mode (Default: clade).")
@click.option("--taxonomy", "-t", type=click.Path(path_type=Path), help="JolTax cache directory.")
@click.option("--exclude-samples", type=click.Path(exists=True, path_type=Path), help="File with sample IDs to exclude.")
@click.option("--min-observed", type=int, default=25, help="Minimum samples taxon must be in (Default: 25).")
@click.option("--min-reads", type=int, default=50, help="Minimum max reads across samples (Default: 50).")
@click.option("--clade", type=int, help="Filter for taxa rooted at this TaxID.")
@click.option("--keep-unclassified", is_flag=True, help="Keep TaxID 0 (unclassified).")
@click.option("--proportions", is_flag=True, help="Output relative proportions instead of raw reads.")
@click.option("--keep-composition", is_flag=True, help="Retain filtered reads as 'Filtered classified' to preserve global total reads. Forces --keep-unclassified.")
def table(input_parquet, output, mode, taxonomy, exclude_samples, min_observed, min_reads, clade, keep_unclassified, proportions, keep_composition):
    """
    Outputs abundance tables with TaxIDs as rows and samples (YYYY_WW) as columns.
    Supports filtering by clade, minimum occupancy, and read depth.
    """
    if keep_composition:
        keep_unclassified = True

    start_time = time.time()
    ctx = click.get_current_context()
    print_header(ctx)
    print_parameters(ctx.params)
    
    try:
        summary = generate_table_logic(
            input_parquet=input_parquet,
            output_file=output,
            mode=mode,
            taxonomy_dir=taxonomy,
            exclude_samples=exclude_samples,
            min_observed=min_observed,
            min_reads=min_reads,
            clade_filter=clade,
            keep_unclassified=keep_unclassified,
            proportions=proportions,
            keep_composition=keep_composition
        )
        summary["status"] = "Success"
        print_footer(start_time, summary)
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)

@main.command(short_help="Extract reads from kraken parquet files into FASTQ.")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--taxonomy", "-t", type=click.Path(exists=True, path_type=Path), required=True, help="JolTax cache directory.")
@click.option("--tax-id", "-i", required=True, help="Comma-separated TaxIDs to extract.")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=Path("."), help="Directory to save FASTQ files.")
@click.option("--mode", "-m", type=click.Choice(["taxon", "clade"]), default="clade", help="Extraction mode (Default: clade).")
@click.option("--combine-samples", is_flag=True, help="Merge all samples into one file per TaxID.")
@click.option("--year-start", type=int, help="Start year for temporal filtering.")
@click.option("--week-start", type=int, help="Start week for temporal filtering.")
@click.option("--year-end", type=int, help="End year for temporal filtering.")
@click.option("--week-end", type=int, help="End week for temporal filtering.")
def extract_reads(input_path, taxonomy, tax_id, output_dir, mode, combine_samples, year_start, week_start, year_end, week_end):
    """
    Streams KRAKEN_PARQUET files and extracts reads into FASTQ format based on 
    TaxID, mode, and temporal window.
    """
    start_time = time.time()
    ctx = click.get_current_context()
    print_header(ctx)
    print_parameters(ctx.params)
    
    # Parse tax_ids
    try:
        tids = [int(tid.strip()) for tid in tax_id.split(",")]
    except ValueError:
        click.secho("Error: --tax-id must be a comma-separated list of integers.", fg="red", err=True)
        sys.exit(1)
        
    try:
        summary = extract_reads_logic(
            input_path=input_path,
            taxonomy_dir=taxonomy,
            tax_ids=tids,
            output_dir=output_dir,
            mode=mode,
            combine_samples=combine_samples,
            year_start=year_start,
            week_start=week_start,
            year_end=year_end,
            week_end=week_end
        )
        summary["status"] = "Success"
        print_footer(start_time, summary)
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)

@main.command(short_help="Annotate a raw abundance table with taxonomy and metadata.")
@click.argument("input_table", type=click.Path(exists=True, path_type=Path))
@click.option("--taxonomy", "-t", type=click.Path(exists=True, path_type=Path), required=True, help="JolTax cache directory.")
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Path to output file (.csv, .tsv, .parquet).")
@click.option("--metadata", "-m", type=click.Path(exists=True, path_type=Path), multiple=True, help="Path to external metadata files (can be used multiple times).")
def annotate_table(input_table, taxonomy, output, metadata):
    """
    Annotates a numeric <RAW_TABLE> with full taxonomic lineages and sorts
    the rows hierarchically. Also computes summary abundance statistics and
    allows joining arbitrary external metadata files.
    """
    start_time = time.time()
    ctx = click.get_current_context()
    print_header(ctx)
    print_parameters(ctx.params)
    
    try:
        summary = annotate_table_logic(
            input_table=input_table,
            taxonomy_dir=taxonomy,
            output_file=output,
            metadata_files=list(metadata) if metadata else []
        )
        summary["status"] = "Success"
        print_footer(start_time, summary)
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)

@main.command(short_help="Show metadata of a SweetBITS parquet file.")
@click.argument("parquet_file", type=click.Path(exists=True, path_type=Path))
def inspect(parquet_file):
    """Prints the global metadata stored in a SweetBITS-generated Parquet file."""
    try:
        metadata = read_parquet_metadata(parquet_file)
        if not metadata:
            click.echo("No SweetBITS metadata found in this file.")
            return
            
        click.secho(f"File: {parquet_file.name}", fg="cyan", bold=True)
        click.echo("-" * 60)
        for key, value in metadata.items():
            display_key = key.replace("_", " ").title()
            click.echo(f"{display_key:20}: {value}")
    except Exception as e:
        click.secho(f"Error reading metadata: {str(e)}", fg="red", err=True)
        sys.exit(1)

@main.command(short_help="Convert Kraken and FASTQ into a KRAKEN_PARQUET.")
@click.argument("kraken_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Path to output Parquet file.")
@click.option("--r1", type=click.Path(exists=True, path_type=Path), help="Path to R1 FASTQ file.")
@click.option("--r2", type=click.Path(exists=True, path_type=Path), help="Path to R2 FASTQ file.")
@click.option("--no-fastq", is_flag=True, help="Create a Skinny Parquet, omitting sequences.")
@click.option("--cores", type=int, help="Number of CPU cores to use (Default: all available).")
def convert_kraken(kraken_file, output, r1, r2, no_fastq, cores):
    """
    Converts Kraken output and FASTQ files into high-performance KRAKEN_PARQUET files.
    """
    start_time = time.time()
    ctx = click.get_current_context()
    print_header(ctx)
    print_parameters(ctx.params)
    
    try:
        summary = convert_kraken_logic(
            kraken_file=kraken_file,
            output_file=output,
            r1_file=r1,
            r2_file=r2,
            no_fastq=no_fastq,
            cores=cores
        )
        summary["status"] = "Success"
        print_footer(start_time, summary)
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
