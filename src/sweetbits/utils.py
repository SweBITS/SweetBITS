"""
sweetbits.utils
Utility functions for parsing and validating SweBITS data structures.
"""

import re
from pathlib import Path
from typing import List, Dict, Any

UNCLASSIFIED_TID = 0
FILTERED_TID = 100000000

import os

def check_write_permission(path: Path):
    """
    Checks if we have write permission for a file or directory.
    If the file exists, checks if it's writable.
    If it doesn't, checks if the parent directory is writable.
    
    Args:
        path : Path to the file or directory.
        
    Raises:
        PermissionError : If write permission is denied.
    """
    if path.exists():
        if not os.access(path, os.W_OK):
            raise PermissionError(f"Permission denied: '{path}' is not writable.")
    else:
        parent = path.parent
        # If parent doesn't exist, we recursively check its parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
            
        if not os.access(parent, os.W_OK):
            raise PermissionError(f"Permission denied: Cannot write to '{path}'. Parent directory '{parent}' is not writable.")

def get_sample_info(filename: str) -> Dict[str, Any]:
    """
    Extracts sample ID and metadata (year, week) from a filename.
    
    Automatically detects if the filename follows the SweBITS pattern
    or should be treated as a generic sample.
    
    Args:
        filename : Name of the file (e.g., 'Ki-2022_20_001.kraken').
        
    Returns:
        A dictionary containing:
        - sample_id     : Base name before all extensions.
        - data_standard : 'SWEBITS' or 'GENERIC'.
        - year          : ISO year (0 if GENERIC).
        - week          : ISO week (0 if GENERIC).
    """
    # Extract base name before any extensions (e.g. sample.kraken.gz -> sample)
    sample_id = filename.split('.')[0]
    try:
        info = parse_sample_id(sample_id)
        return {
            "sample_id": sample_id,
            "data_standard": "SWEBITS",
            "year": info["year"],
            "week": info["week"]
        }
    except ValueError:
        return {
            "sample_id": sample_id,
            "data_standard": "GENERIC",
            "year": 0,
            "week": 0
        }

def parse_sample_id(sample_id: str) -> Dict[str, Any]:
    """
    Parses a SweBITS sample ID into its components and validates the format.
    
    Supported formats:
    - Ki-YYYY_WW_ZZZ (Kiruna)
    - Lj-YYYY_WW_ZZZ (Ljungbyhed)
    
    Args:
        sample_id : The string ID to parse (e.g., 'Ki-2022_20_001').

    Returns:
        A dictionary containing:
        - site      : 'Kiruna' or 'Ljungbyhed'
        - year      : int
        - week      : int
        - suffix    : str (the 1-3 digit ZZZ part)
        - sample_id : original sample ID
    
    Raises:
        ValueError  : If the sample_id format is invalid or week is out of range.
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

    Args:
        file_path : Path to the text file.

    Returns:
        A list of unique Sample ID strings extracted from the file.

    Raises:
        FileNotFoundError : If the file path does not exist.
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
