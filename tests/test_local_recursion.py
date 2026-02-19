"""Test local/nested recursive function serialization."""

import sys
sys.path.insert(0, "tests")

from nnsight.intervention.serialization import dumps, loads


def outer():
    """Factory that returns a local recursive function."""
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)
    return factorial


def test_local_recursive_function():
    """Test that a local recursive function can be serialized."""
    local_factorial = outer()

    print(f"\n=== Test local/nested recursive function ===")
    print(f"Original: local_factorial(5) = {local_factorial(5)}")
    print(f"qualname: {local_factorial.__qualname__}")

    data = dumps(local_factorial)
    print(f"Serialized: {len(data)} bytes")

    restored = loads(data)
    print(f"Restored: restored(5) = {restored(5)}")

    assert restored(5) == 120
    assert restored(10) == 3628800
    print("SUCCESS!")


if __name__ == "__main__":
    test_local_recursive_function()
