import re
from typing import Dict, Any

def parse_sample_id(sample_id: str) -> Dict[str, Any]:
    """
    Parses a SweBITS sample ID into its components.
    
    Supported formats:
    - Ki-YYYY_WW_ZZZ (Kiruna)
    - Lj-YYYY_WW_ZZZ (Ljungbyhed)
    
    Returns a dictionary with:
    - site: 'Kiruna' or 'Ljungbyhed'
    - year: int
    - week: int
    - suffix: str (the ZZZ part)
    - sample_id: original sample ID
    
    Raises:
        ValueError: If the sample_id format is invalid.
    """
    pattern = r"^(Ki|Lj)-(\d{4})_(\d{1,2})_(.+)$"
    match = re.match(pattern, sample_id)
    
    if not match:
        raise ValueError(f"Invalid sample ID format: {sample_id}")
    
    site_code, year, week, suffix = match.groups()
    
    site_map = {
        "Ki": "Kiruna",
        "Lj": "Ljungbyhed"
    }
    
    return {
        "site": site_map[site_code],
        "year": int(year),
        "week": int(week),
        "suffix": suffix,
        "sample_id": sample_id
    }
