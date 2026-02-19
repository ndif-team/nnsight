"""Test lambda function serialization.

Tests various lambda serialization scenarios including:
- Simple lambdas
- Lambdas with closures
- Multiple lambdas on same line
- Nested lambdas (lambda returning lambda)
- Lambdas with lambda default values
- Multi-line lambdas
"""

import sys
sys.path.insert(0, "tests")

import pytest
from nnsight.intervention.serialization import dumps, loads


class TestLambdaSerialization:
    """Tests for lambda function serialization."""

    def test_simple_lambda(self):
        """Test simple lambda serialization."""
        f = lambda x: x * 2

        data = dumps(f)
        restored = loads(data)

        assert restored(5) == 10

    def test_lambda_with_closure(self):
        """Test lambda with captured variable."""
        multiplier = 3
        f = lambda x: x * multiplier

        data = dumps(f)
        restored = loads(data)

        assert restored(5) == 15

    def test_lambda_with_multiple_args(self):
        """Test lambda with multiple arguments."""
        f = lambda x, y, z: x + y + z

        data = dumps(f)
        restored = loads(data)

        assert restored(1, 2, 3) == 6

    def test_lambda_with_default_args(self):
        """Test lambda with default arguments including dict with colons."""
        f = lambda x, y=10: x + y

        data = dumps(f)
        restored = loads(data)

        assert restored(5) == 15
        assert restored(5, 20) == 25

        # Dict default with colons - tests that depth tracking works
        g = lambda x={'a': 1, 'b': 2}: x['a'] + x['b']

        data = dumps(g)
        restored = loads(data)

        assert restored() == 3
        assert restored({'a': 10, 'b': 20}) == 30

    def test_nested_lambda(self):
        """Test lambda returning lambda."""
        f = lambda x: lambda y: x + y

        data = dumps(f)
        restored = loads(data)

        add_5 = restored(5)
        assert add_5(3) == 8

    def test_lambda_as_default_value(self):
        """Test lambda with another lambda as default argument value."""
        f = lambda x=lambda: 1: x()

        data = dumps(f)
        restored = loads(data)

        assert restored() == 1
        assert restored(lambda: 42) == 42

    def test_multiple_lambdas_same_line(self):
        """Test multiple independent lambdas defined on the same line."""
        f, g = lambda x: x * 2, lambda x: x + 1

        data_f = dumps(f)
        data_g = dumps(g)

        restored_f = loads(data_f)
        restored_g = loads(data_g)

        assert restored_f(5) == 10
        assert restored_g(5) == 6

    def test_multiline_lambda_with_neighbors(self):
        """Test multi-line lambda sharing lines with other lambdas."""
        # fmt: off
        a, b, c, d = lambda x: x + 1, lambda y: (
            y * 2 +
            y * 3
        ), (lambda z:
            z - 1), lambda w: w * 2
        # fmt: on

        data_a = dumps(a)
        data_b = dumps(b)
        data_c = dumps(c)
        data_d = dumps(d)

        restored_a = loads(data_a)
        restored_b = loads(data_b)
        restored_c = loads(data_c)
        restored_d = loads(data_d)

        assert restored_a(10) == 11
        assert restored_b(10) == 50  # 10*2 + 10*3
        assert restored_c(10) == 9
        assert restored_d(10) == 20

    def test_lambda_in_list(self):
        """Test serializing lambdas in a list."""
        funcs = [lambda x: x + 1, lambda x: x * 2, lambda x: x ** 2]

        data = dumps(funcs)
        restored = loads(data)

        assert restored[0](5) == 6
        assert restored[1](5) == 10
        assert restored[2](5) == 25

    def test_lambda_in_dict(self):
        """Test serializing lambdas as dict values."""
        ops = {
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
        }

        data = dumps(ops)
        restored = loads(data)

        assert restored['add'](10, 3) == 13
        assert restored['sub'](10, 3) == 7
        assert restored['mul'](10, 3) == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
