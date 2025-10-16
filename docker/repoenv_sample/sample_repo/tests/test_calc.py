import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import app.calc as calc


def test_addition():
    # This expectation fails until the agent fixes the bug inside app.calc.add.
    assert calc.add(2, 2) == 4


def test_multiplication():
    assert calc.mul(3, 7) == 21
