import pytest
from sweetbits.utils import parse_sample_id

def test_parse_sample_id_kiruna():
    sample_id = "Ki-1974_02_ABC"
    result = parse_sample_id(sample_id)
    assert result["site"] == "Kiruna"
    assert result["year"] == 1974
    assert result["week"] == 2
    assert result["suffix"] == "ABC"
    assert result["sample_id"] == sample_id

def test_parse_sample_id_ljungbyhed():
    sample_id = "Lj-2022_52_XYZ"
    result = parse_sample_id(sample_id)
    assert result["site"] == "Ljungbyhed"
    assert result["year"] == 2022
    assert result["week"] == 52
    assert result["suffix"] == "XYZ"
    assert result["sample_id"] == sample_id

def test_parse_sample_id_invalid():
    with pytest.raises(ValueError, match="Invalid sample ID format"):
        parse_sample_id("Invalid-Format")

def test_parse_sample_id_missing_parts():
    with pytest.raises(ValueError):
        parse_sample_id("Ki-1974_02")
