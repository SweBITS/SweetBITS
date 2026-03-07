"""
sweetbits.utils
Utility functions for parsing and validating SweBITS data structures.
"""

import re
from pathlib import Path
from typing import Dict, Any, List

def parse_sample_id(sample_id: str) -> Dict[str, Any]:
    """
    Parses a SweBITS sample ID into its components and validates the format.
    
    Supported formats:
    - Ki-YYYY_WW_ZZZ (Kiruna)
    - Lj-YYYY_WW_ZZZ (Ljungbyhed)
    
    The format strictly requires:
    - Site: 'Ki' or 'Lj'
    - Year: 4 digits
    - Week: 1 or 2 digits (validated to range 1-53)
    - Suffix: 1 to 3 digits
    
    Args:
        sample_id: The string ID to parse (e.g., 'Ki-2022_20_001', 'Lj_2013_1_142').

    Returns:
        A dictionary containing:
        - site: 'Kiruna' or 'Ljungbyhed'
        - year: int
        - week: int
        - suffix: str (the 1-3 digit ZZZ part)
        - sample_id: original sample ID
    
    Raises:
        ValueError: If the sample_id format is invalid or week is out of range.
    """
    pattern = r"^(Ki|Lj)[-_](\d{4})[-_](\d{1,2})[-_](\d{1,3})$"
    match = re.match(pattern, sample_id)
    
    if not match:
        raise ValueError(
            f"Invalid sample ID format: '{sample_id}'. "
            "Expected 'Ki-YYYY_WW_ZZZ' or 'Lj-YYYY_WW_ZZZ' with numeric components (1-3 digit suffix)."
        )
    
    site_code, year_str, week_str, suffix = match.groups()
    
    year = int(year_str)
    week = int(week_str)
    
    if not (1 <= week <= 53):
        raise ValueError(f"Invalid ISO week in sample ID: {week}. Must be between 1 and 53.")
    
    site_map = {
        "Ki": "Kiruna",
        "Lj": "Ljungbyhed"
    }
    
    return {
        "site": site_map[site_code],
        "year": year,
        "week": week,
        "suffix": suffix,
        "sample_id": sample_id
    }

def load_sample_id_list(file_path: Path) -> List[str]:
    """
    Reads a list of Sample IDs from a text file (one ID per line).
    
    Handles:
    - Whitespace stripping.
    - Skipping empty lines.
    - Skipping comment lines (starting with #).
    - Basic validation (ensures IDs are strings).

    Args:
        file_path: Path to the text file.

    Returns:
        A list of unique Sample ID strings.
    """
    ids = []
    if not file_path.exists():
        raise FileNotFoundError(f"Sample ID list file not found: {file_path}")
        
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
            
    # Remove duplicates but preserve order
    seen = set()
    return [x for x in ids if not (x in seen or seen.add(x))]
