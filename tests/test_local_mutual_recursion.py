"""Test local mutual recursion serialization.

This tests the harder case where mutually recursive functions are defined
locally (inside another function), not at module level.
"""

import sys
sys.path.insert(0, "tests")

import pytest
from nnsight.intervention.serialization import dumps, loads


def test_local_mutual_recursion():
    """Test mutually recursive local functions.

    This tests the harder case where mutually recursive functions reference
    each other through closures. The fix defers ALL function values in closures
    to the state setter, allowing pickle's memo to break the cycle.
    """

    def is_even(n):
        if n == 0:
            return True
        return is_odd(n - 1)

    def is_odd(n):
        if n == 0:
            return False
        return is_even(n - 1)

    print(f"\n=== Test local mutual recursion ===")
    print(f"is_even(4) = {is_even(4)}")  # True
    print(f"is_odd(4) = {is_odd(4)}")    # False

    # Serialize both together
    print("\n=== Serialize both together ===")
    data = dumps((is_even, is_odd))
    print(f"Serialized: {len(data)} bytes")

    restored_even, restored_odd = loads(data)
    print(f"restored_even(4) = {restored_even(4)}")
    print(f"restored_odd(4) = {restored_odd(4)}")

    assert restored_even(4) is True
    assert restored_even(5) is False
    assert restored_odd(4) is False
    assert restored_odd(5) is True
    print("SUCCESS!")


if __name__ == "__main__":
    test_local_mutual_recursion()
