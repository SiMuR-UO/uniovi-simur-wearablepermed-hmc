import pytest

from wearablepermed_hmc.converter import converter, main

__author__ = "Miguel Angel Salinas Gancedo"
__copyright__ = "Miguel Angel Salinas Gancedo"
__license__ = "MIT"

def test_converter():
    """API Tests"""
    assert converter('MATA00.BIN', 'MATA00.xlsx') == 0
    with pytest.raises(AssertionError):
        converter('MATA00.BIN', 'MATA00.xlsx')


#def test_main(capsys):
#    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
#    main(["7"])
#    captured = capsys.readouterr()
#    assert "The 7-th Fibonacci number is 13" in captured.out
