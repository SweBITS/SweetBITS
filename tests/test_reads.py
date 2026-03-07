import pytest
from sweetbits.reads import format_short_name, is_in_temporal_range

def test_format_short_name_multiple_words():
    assert format_short_name("Homo sapiens") == "HomSap"
    assert format_short_name("Candidatus Liberibacter asiaticus") == "CanLib"

def test_format_short_name_single_word():
    assert format_short_name("Bacteria") == "Bacteria"
    assert format_short_name("root") == "root"

def test_is_in_temporal_range():
    # Simple range
    assert is_in_temporal_range(2013, 10, 2013, 1, 2013, 20) is True
    assert is_in_temporal_range(2013, 25, 2013, 1, 2013, 20) is False
    
    # Range across years
    assert is_in_temporal_range(2013, 50, 2013, 40, 2014, 5) is True
    assert is_in_temporal_range(2014, 2, 2013, 40, 2014, 5) is True
    assert is_in_temporal_range(2014, 10, 2013, 40, 2014, 5) is False
    
    # Open ended
    assert is_in_temporal_range(2020, 1, year_start=2019) is True
    assert is_in_temporal_range(2018, 1, year_start=2019) is False
