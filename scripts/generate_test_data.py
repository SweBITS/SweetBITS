from pathlib import Path
from sweetbits.testing import generate_mock_kraken_parquet, generate_mock_report_parquet

def main():
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    sample_id = "Lj-2022_20_001"
    
    print(f"Generating mock Kraken Parquet for {sample_id}...")
    generate_mock_kraken_parquet(
        sample_id=sample_id,
        num_reads=100,
        output_path=test_data_dir / f"{sample_id}.kraken.parquet"
    )
    
    print("Generating mock Report Parquet...")
    generate_mock_report_parquet(
        sample_ids=[sample_id, "Lj-2022_20_002"],
        output_path=test_data_dir / "merged_reports.parquet"
    )
    
    print(f"Done. Test data generated in {test_data_dir}/")

if __name__ == "__main__":
    main()
