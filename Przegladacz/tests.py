from backend import get_float_from_string, get_integer_from_string

import pytest


@pytest.mark.parametrize(
    "string_input, expected_output",
    [
        ("-13.8", -13.8),
        ("142", 142),
        ("abc", None),
        ("13.334.23", None),
        ("1,100,100.13", 1100100.13),
    ],
)
def test_get_float_from_string(string_input, expected_output):
    assert get_float_from_string(string_input) == expected_output


@pytest.mark.parametrize(
    "string_input, expected_output",
    [
        ("-13.8", 138),
        ("142", 142),
        ("abc", None),
        ("13.334.23", 1333423),
        ("1,100,100.13", 110010013),
    ],
)
def test_get_float_from_string(string_input, expected_output):
    asserfrom backend import get_float_from_string, get_integer_from_string

import pytest


@pytest.mark.parametrize(
    "string_input, expected_output",
    [
        ("-13.8", -13.8),
        ("142", 142),
        ("abc", None),
        ("13.334.23", None),
        ("1,100,100.13", 1100100.13),
    ],
)
def test_get_float_from_string(string_input, expected_output):
    assert get_float_from_string(string_input) == expected_output


@pytest.mark.parametrize(
    "string_input, expected_output",
    [
        ("-13.8", 138),
        ("142", 142),
        ("abc", None),
        ("13.334.23", 1333423),
        ("1,100,100.13", 110010013),
    ],
)
def test_get_float_from_string(string_input, expected_output):
    assert get_integer_from_string(string_input) == expected_output
