"""Test mutual recursion serialization.

This tests module-level mutually recursive functions where the functions
reference each other through globals (not closures).
"""

import sys
sys.path.insert(0, "tests")

from nnsight.intervention.serialization import dumps, loads


def is_even(n):
    if n == 0:
        return True
    return is_odd(n - 1)


def is_odd(n):
    if n == 0:
        return False
    return is_even(n - 1)


def test_mutual_recursion_single():
    """Test serializing one function that calls another."""
    # Verify original functions work
    assert is_even(4) is True
    assert is_odd(4) is False

    # Serialize is_even (which calls is_odd)
    data = dumps(is_even)
    restored = loads(data)

    assert restored(4) is True
    assert restored(5) is False


def test_mutual_recursion_together():
    """Test serializing both mutually recursive functions together."""
    data = dumps((is_even, is_odd))
    restored_even, restored_odd = loads(data)

    assert restored_even(4) is True
    assert restored_even(5) is False
    assert restored_odd(4) is False
    assert restored_odd(5) is True


if __name__ == "__main__":
    test_mutual_recursion_single()
    print("test_mutual_recursion_single: PASSED")
    test_mutual_recursion_together()
    print("test_mutual_recursion_together: PASSED")
