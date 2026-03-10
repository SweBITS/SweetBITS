import pytest
import os
from pathlib import Path
from sweetbits.tables import generate_table_logic
from sweetbits.reports import gather_reports_logic

def test_permission_check_early(tmp_path):
    # Create a read-only directory
    ro_dir = tmp_path / "read_only"
    ro_dir.mkdir()
    output = ro_dir / "test.parquet"
    
    # Change permissions to 555 (read and execute, no write)
    os.chmod(ro_dir, 0o555)
    
    try:
        # Should raise PermissionError before doing work
        with pytest.raises(PermissionError, match="Permission denied"):
            gather_reports_logic(Path("."), output)
            
        with pytest.raises(PermissionError, match="Permission denied"):
            generate_table_logic(Path("test.parquet"), output, taxonomy_dir=Path("tax"))
    finally:
        # Restore permissions so tmp_path can be cleaned up
        os.chmod(ro_dir, 0o755)

def test_early_keep_composition_validation(tmp_path):
    output = tmp_path / "test.tsv"
    # Should raise ValueError before any file access
    with pytest.raises(ValueError, match="not mathematically valid for 'clade' mode"):
        generate_table_logic(
            input_parquet=Path("non_existent.parquet"),
            output_file=output,
            mode="clade",
            keep_composition=True,
            taxonomy_dir=Path("tax")
        )
