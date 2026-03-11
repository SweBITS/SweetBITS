import click
import time
import sys
import os
try:
    import resource
except ImportError:
    resource = None
from pathlib import Path
from datetime import datetime
from sweetbits import __version__
from sweetbits.reports import gather_reports_logic
from sweetbits.tables import generate_table_logic
from sweetbits.reads import extract_reads_logic
from sweetbits.annotate import annotate_table_logic
from sweetbits.convert import convert_kraken_logic

def print_splash():
    """Prints the stylish ASCII logo and developer information."""
    click.echo("", err=True)
    click.secho(r"  ____                     _   ____  ___ _____ ____  ", fg="bright_cyan", bold=True, err=True)
    click.secho(r" / ___|_      _____  ___ _| |_| __ )|_ _|_   _/ ___| ", fg="bright_cyan", bold=True, err=True)
    click.secho(r" \___ \ \ /\ / / _ \/ _ \ ____|  _ \ | |  | | \___ \  ", fg="cyan", bold=True, err=True)
    click.secho(r"  ___) \ V  V /  __/  __/_| |_| |_) || |  | |  ___) | ", fg="cyan", bold=True, err=True)
    click.secho(r" |____/ \_/\_/ \___|\_____|\__|____/|___| |_| |____/  ", fg="cyan", bold=True, err=True)
    click.echo("", err=True)
    click.secho(" A suite of sweet command-line tools for Kraken 2", fg="bright_white", err=True)
    click.secho(" derived data and the SweBITS project.", fg="bright_white", err=True)
    click.echo("", err=True)
    click.echo(click.style(" Developer: ", fg="bright_black") + click.style("Daniel Svensson for SweBITS", fg="yellow"), err=True)
    click.echo(click.style(" GitHub:    ", fg="bright_black") + click.style("https://github.com/SweBITS/SweetBITS", fg="bright_blue"), err=True)
    click.echo(click.style(" Version:   ", fg="bright_black") + click.style(f"v{__version__}", fg="yellow"), err=True)
    click.echo("", err=True)
    click.echo("-" * 60, err=True)

def print_invocation_info():
    """Prints technical details about the current execution."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    click.echo(f"{'Start time':20}: {now}", err=True)
    click.echo(f"{'SweetBITS version':20}: {__version__}", err=True)
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
            
    # Report Peak Memory (Unix only)
    if resource:
        usage_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        usage_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        total_units = usage_self + usage_child
        
        # On Linux ru_maxrss is in KB, on macOS it is in Bytes
        divisor = 1024 if sys.platform != 'darwin' else 1024 * 1024
        total_mb = total_units / divisor
        click.echo(f"{'Peak Memory':20}: {total_mb:.2f} MB", err=True)

    click.echo(f"{'Time elapsed':20}: {elapsed:.2f}s", err=True)
    click.echo(f"{'End time':20}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", err=True)
    click.echo("", err=True)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.option('--version', is_flag=True, help="Show version and splash screen.")
@click.pass_context
def main(ctx, version):
    if version:
        print_splash()
        ctx.exit()

    is_help = any(arg in sys.argv for arg in ['-h', '--help'])
    # Need to check subcommands recursively for splash suppression, but keeping simple for now
    # We will just suppress it if we are deeper than root.
    has_subcommand = len(sys.argv) > 1 and sys.argv[1] not in ['-h', '--help', '--version']

    if ctx.invoked_subcommand is None:
        print_splash()
        click.echo(ctx.get_help())
        ctx.exit()
    elif is_help and not has_subcommand:
        print_splash()

@main.group(short_help="Ingest data into SweetBITS Parquet format.")
def collect():
    """Commands to parse and ingest output files from taxonomic classifiers."""
    pass

@collect.group(short_help="Ingest Kraken 2 output files.")
def kraken():
    """Commands specifically for parsing Kraken 2 reports and read classifications."""
    pass

@main.group(short_help="Generate usable outputs from SweetBITS data.")
def produce():
    """Commands to generate outputs like abundance tables and FASTQ files."""
    pass

@kraken.command(name="reports", short_help="Merge Kraken reports into a single Parquet file (<REPORTS_PARQUET>).")
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Path to output Parquet file.")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories (Default: True).")
@click.option("--include", "-i", default="*.report", help="Pattern to match report files (Default: *.report).")
@click.option("--cores", type=int, help="Number of CPU cores to use (Default: all available).")
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists.")
def collect_kraken_reports(directory, output, recursive, include, cores, overwrite):
    """
    Finds and merges multiple 8-column Kraken reports into a single Parquet file. 
    Output is a <REPORTS_PARQUET> with an accompanying JSON metadata companion file.
    """
    start_time = time.time()
    ctx = click.get_current_context()
    print_splash()
    print_invocation_info()
    print_parameters(ctx.params)
    
    try:
        summary = gather_reports_logic(
            input_dir=directory,
            output_file=output,
            recursive=recursive,
            include_pattern=include,
            cores=cores,
            overwrite=overwrite
        )
        summary["status"] = "Success"
        summary["output_file"] = str(output)
        print_footer(start_time, summary)
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)

@produce.command(name="table", short_help="Generate abundance tables (<RAW_TABLE>) from merged reports (<REPORTS_PARQUET>).")
@click.argument("input_parquet", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=False, help="Path to output file (.csv, .tsv, .parquet). Required unless --dry-run is used.")
@click.option("--mode", "-m", type=click.Choice(["taxon", "clade", "canonical"]), default="clade", help="Abundance mode (Default: clade).")
@click.option("--taxonomy", "-t", type=click.Path(path_type=Path), required=True, help="JolTax cache directory.")
@click.option("--exclude-samples", type=click.Path(exists=True, path_type=Path), help="File with sample IDs to exclude.")
@click.option("--min-observed", type=int, default=25, help="Minimum samples taxon must be in (Default: 25).")
@click.option("--min-reads", type=int, default=50, help="Minimum max reads across samples (Default: 50).")
@click.option("--clade", "clade_filter", type=int, help="Filter for taxa rooted at this TaxID.")
@click.option("--keep-unclassified", is_flag=True, help="Keep TaxID 0 (unclassified).")
@click.option("--proportions", is_flag=True, help="Output relative proportions instead of raw reads.")
@click.option("--keep-filtered", is_flag=True, help="Retain filtered reads as 'Filtered classified' to preserve global total reads. Forces --keep-unclassified.")
@click.option("--cores", type=int, help="Number of CPU cores to use (Default: all available).")
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists.")
@click.option("--dry-run", is_flag=True, help="Print an audit report of filtering retention without saving the file.")
def produce_table(input_parquet, output, mode, taxonomy, exclude_samples, min_observed, min_reads, clade_filter, keep_unclassified, proportions, keep_filtered, cores, overwrite, dry_run):
    """
    Outputs abundance tables with TaxIDs as rows and samples as columns.
    Supports filtering by clade, minimum occupancy, and read depth. Output is
    <RAW_TABLE>. An audit report is printed to the terminal summarizing data 
    retention after filtering.
    """
    if not output and not dry_run:
        raise click.UsageError("Missing option '--output' / '-o'.")
        
    # Create a dummy path for dry-run if not provided to pass type checks
    output_path = output if output else Path("dry_run_output.tmp")

    if keep_filtered:
        keep_unclassified = True

    start_time = time.time()
    ctx = click.get_current_context()
    
    # Force sync parameters for display and logic
    if keep_filtered:
        ctx.params["keep_unclassified"] = True
    
    # For display, we use the context params dictionary
    params_for_display = ctx.params.copy()
    
    print_splash()
    print_invocation_info()
    print_parameters(params_for_display)
    
    try:
        summary = generate_table_logic(
            input_parquet=input_parquet,
            output_file=output_path,
            mode=mode,
            taxonomy_dir=ctx.params.get("taxonomy"),
            exclude_samples=exclude_samples,
            min_observed=min_observed,
            min_reads=min_reads,
            clade_filter=clade_filter,
            keep_unclassified=ctx.params.get("keep_unclassified"),
            proportions=proportions,
            keep_filtered=keep_filtered,
            cores=cores,
            overwrite=overwrite,
            dry_run=dry_run
        )
        summary["status"] = "Success"
        print_footer(start_time, summary)
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)

@produce.command(name="reads", short_help="Extract reads from <KRAKEN_PARQUET> files into FASTQ.")
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
@click.option("--cores", type=int, help="Number of CPU cores to use (Default: all available).")
@click.option("--overwrite", is_flag=True, help="Overwrite output files if they exist.")
def produce_reads(input_path, taxonomy, tax_id, output_dir, mode, combine_samples, year_start, week_start, year_end, week_end, cores, overwrite):
    """
    Extracts reads or read IDs from <KRAKEN_PARQUET> files.
    """
    start_time = time.time()
    ctx = click.get_current_context()
    print_splash()
    print_invocation_info()
    print_parameters(ctx.params)

    try:
        t_ids = [int(tid.strip()) for tid in tax_id.split(",")]
    except ValueError:
        click.secho("Error: --tax-id must be a comma-separated list of integers.", fg="red", err=True)
        sys.exit(1)

    try:
        summary = extract_reads_logic(
            input_path=input_path,
            taxonomy_dir=taxonomy,
            tax_ids=t_ids,
            output_dir=output_dir,
            mode=mode,
            combine_samples=combine_samples,
            year_start=year_start,
            week_start=week_start,
            year_end=year_end,
            week_end=week_end,
            cores=cores,
            overwrite=overwrite
        )
        summary["status"] = "Success"
        print_footer(start_time, summary)
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)

@main.command(name="annotate", short_help="Annotate a <RAW_TABLE> with taxonomy and metadata.")
@click.argument("input_table", type=click.Path(exists=True, path_type=Path))
@click.option("--taxonomy", "-t", type=click.Path(exists=True, path_type=Path), required=True, help="JolTax cache directory.")
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Path to output file (.csv, .tsv, .parquet).")
@click.option("--metadata", "-m", type=click.Path(exists=True, path_type=Path), multiple=True, help="Path to external metadata files (can be used multiple times).")
@click.option("--sort-order", default="alphabetical", type=click.Choice(["alphabetical", "dfs"]), help="Row sorting order (alphabetical or abundance-weighted DFS).")
@click.option("--cores", type=int, help="Number of CPU cores to use (Default: all available).")
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists.")
def annotate(input_table, taxonomy, output, metadata, sort_order, cores, overwrite):
    """
    Annotates a numeric <RAW_TABLE> with full taxonomic lineages and sorts
    the rows. Also computes summary abundance statistics and
    allows joining arbitrary external metadata files.
    """
    start_time = time.time()
    ctx = click.get_current_context()
    print_splash()
    print_invocation_info()
    print_parameters(ctx.params)
    
    try:
        summary = annotate_table_logic(
            input_table=input_table,
            taxonomy_dir=taxonomy,
            output_file=output,
            metadata_files=list(metadata) if metadata else [],
            sort_order=sort_order,
            cores=cores,
            overwrite=overwrite
        )
        summary["status"] = "Success"
        print_footer(start_time, summary)
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)

@main.command(name="inspect", short_help="Show metadata of a SweetBITS file via its JSON companion.")
@click.argument("target_file", type=click.Path(exists=True, path_type=Path))
def inspect(target_file):
    """
    Prints the global provenance metadata stored in a SweetBITS-generated JSON companion file.
    Works for any SweetBITS output (e.g., Parquet, CSV, TSV).
    """
    from sweetbits.metadata import read_companion_metadata

    print_splash()
    try:
        # If the user pointed directly to the json file, use it as the base
        if target_file.suffix == ".json":
            data_file = target_file.with_name(target_file.name[:-5])
        else:
            data_file = target_file

        metadata = read_companion_metadata(data_file)
        
        if not metadata:
            click.secho(
                f"Error: Missing companion metadata file '{data_file.name}.json'.\n"
                "SweetBITS requires this JSON file to verify data integrity and provenance.", 
                fg="red", err=True
            )
            sys.exit(1)

        click.echo(click.style("Inspecting: ", fg="cyan") + click.style(f"{data_file.name}.json", fg="cyan", bold=True))
        click.echo("-" * 55)
        for key, value in metadata.items():
            display_key = key.replace("_", " ").title().replace("Sweetbits", "SweetBITS")
            click.echo(f"{display_key:20}: {value}")
    except Exception as e:
        click.secho(f"Error reading metadata: {str(e)}", fg="red", err=True)
        sys.exit(1)

@kraken.command(name="classifications", short_help="Convert a Kraken 2 and R1/R2 FASTQ files into a <KRAKEN_PARQUET>.")
@click.argument("kraken_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Path to output Parquet file.")
@click.option("--r1", type=click.Path(exists=True, path_type=Path), help="Path to R1 FASTQ file.")
@click.option("--r2", type=click.Path(exists=True, path_type=Path), help="Path to R2 FASTQ file.")
@click.option("--cores", type=int, help="Number of CPU cores to use (Default: all available).")
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists.")
def collect_kraken_classifications(kraken_file, output, r1, r2, cores, overwrite):
    """
    Converts a Kraken 2 read-by-read output file (and optionally associated FASTQ files) into
    a high-performance Parquet file. Output is <KRAKEN_PARQUET>.
    """
    start_time = time.time()
    ctx = click.get_current_context()
    print_splash()
    print_invocation_info()
    print_parameters(ctx.params)
    
    try:
        summary = convert_kraken_logic(
            kraken_file=kraken_file,
            output_file=output,
            r1_file=r1,
            r2_file=r2,
            cores=cores,
            overwrite=overwrite
        )
        summary["status"] = "Success"
        print_footer(start_time, summary)
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
