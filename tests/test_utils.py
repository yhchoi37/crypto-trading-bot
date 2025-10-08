import pytest
from src.utils import calculate_percentage_change, safe_divide


def test_calculate_percentage_change_positive():
    assert pytest.approx(calculate_percentage_change(100, 110), rel=1e-6) == 10.0


def test_calculate_percentage_change_zero_old():
    assert calculate_percentage_change(0, 50) == 0.0


def test_safe_divide():
    assert safe_divide(10, 2) == 5
    assert safe_divide(1, 0, default=999) == 999
