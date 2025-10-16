"""Deliberately buggy arithmetic helpers used for RepoEnv integration tests."""

from __future__ import annotations


def add(lhs: float, rhs: float) -> float:
    """Return the sum of two numbers.

    This implementation intentionally contains a defect so the agent has
    something to repair once the test suite is executed inside the container.
    """

    return lhs + rhs


def mul(lhs: float, rhs: float) -> float:
    """Return the product of two numbers."""

    return lhs * rhs
