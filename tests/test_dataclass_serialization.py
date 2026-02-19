"""Test dataclass serialization.

Tests various dataclass serialization scenarios including:
- Simple dataclasses
- Dataclasses with default values
- Dataclasses with default_factory
- Dataclasses with frozen/eq/order parameters
- Nested dataclasses
- Dataclasses with type annotations
- Dataclasses with user-defined methods
- Dataclass instances
- Dataclass inheritance
"""

import sys
sys.path.insert(0, "tests")

import pytest
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from nnsight.intervention.serialization import dumps, loads


class TestDataclassSerialization:
    """Tests for dataclass class serialization."""

    def test_simple_dataclass(self):
        """Test simple dataclass serialization."""
        @dataclass
        class Point:
            x: int
            y: int

        data = dumps(Point)
        restored = loads(data)

        # Check class can be instantiated
        p = restored(10, 20)
        assert p.x == 10
        assert p.y == 20

    def test_dataclass_with_defaults(self):
        """Test dataclass with default values."""
        @dataclass
        class Config:
            name: str
            value: int = 42
            enabled: bool = True

        data = dumps(Config)
        restored = loads(data)

        # Test with all args
        c1 = restored("test", 100, False)
        assert c1.name == "test"
        assert c1.value == 100
        assert c1.enabled is False

        # Test with defaults
        c2 = restored("default_test")
        assert c2.name == "default_test"
        assert c2.value == 42
        assert c2.enabled is True

    def test_dataclass_with_default_factory(self):
        """Test dataclass with default_factory."""
        @dataclass
        class Container:
            items: List[int] = field(default_factory=list)
            metadata: Dict[str, Any] = field(default_factory=dict)

        data = dumps(Container)
        restored = loads(data)

        # Each instance should get its own list/dict
        c1 = restored()
        c2 = restored()

        c1.items.append(1)
        c1.metadata["key"] = "value"

        assert c1.items == [1]
        assert c2.items == []  # c2 should have its own empty list
        assert c1.metadata == {"key": "value"}
        assert c2.metadata == {}

    def test_frozen_dataclass(self):
        """Test frozen dataclass serialization."""
        @dataclass(frozen=True)
        class ImmutablePoint:
            x: int
            y: int

        data = dumps(ImmutablePoint)
        restored = loads(data)

        p = restored(5, 10)
        assert p.x == 5
        assert p.y == 10

        # Verify frozen behavior
        with pytest.raises(AttributeError):
            p.x = 100

    def test_dataclass_with_eq_and_order(self):
        """Test dataclass with eq and order parameters."""
        @dataclass(eq=True, order=True)
        class SortableItem:
            priority: int
            name: str

        data = dumps(SortableItem)
        restored = loads(data)

        item1 = restored(1, "alpha")
        item2 = restored(2, "beta")
        item3 = restored(1, "gamma")

        # Test equality
        assert item1 != item2
        assert item1 == restored(1, "alpha")

        # Test ordering - sorts by priority first, then name
        assert item1 < item2
        assert item2 > item1
        # item1 (1, "alpha") < item3 (1, "gamma") < item2 (2, "beta")
        assert sorted([item2, item1, item3]) == [item1, item3, item2]

    def test_dataclass_with_unsafe_hash(self):
        """Test dataclass with unsafe_hash=True."""
        @dataclass(unsafe_hash=True)
        class HashableItem:
            id: int
            name: str

        data = dumps(HashableItem)
        restored = loads(data)

        item = restored(42, "test")

        # Should be hashable
        hash_value = hash(item)
        assert isinstance(hash_value, int)

        # Can be used in sets/dicts
        s = {item}
        assert item in s

    def test_nested_dataclass(self):
        """Test nested dataclass serialization."""
        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            inner: Inner
            name: str

        data = dumps(Outer)
        restored_outer = loads(data)

        # Also need to serialize Inner for reconstruction
        inner_data = dumps(Inner)
        restored_inner = loads(inner_data)

        inner = restored_inner(100)
        outer = restored_outer(inner, "test")

        assert outer.inner.value == 100
        assert outer.name == "test"

    def test_dataclass_with_optional_types(self):
        """Test dataclass with Optional type annotations."""
        @dataclass
        class NullableData:
            required: str
            optional: Optional[int] = None
            also_optional: Optional[List[str]] = None

        data = dumps(NullableData)
        restored = loads(data)

        # Test with None values
        d1 = restored("required_value")
        assert d1.required == "required_value"
        assert d1.optional is None
        assert d1.also_optional is None

        # Test with actual values
        d2 = restored("required", 42, ["a", "b"])
        assert d2.optional == 42
        assert d2.also_optional == ["a", "b"]

    def test_dataclass_with_user_methods(self):
        """Test dataclass with user-defined methods."""
        @dataclass
        class Calculator:
            x: int
            y: int

            def add(self) -> int:
                return self.x + self.y

            def multiply(self) -> int:
                return self.x * self.y

            @property
            def sum_and_product(self) -> tuple:
                return (self.add(), self.multiply())

        data = dumps(Calculator)
        restored = loads(data)

        calc = restored(3, 4)
        assert calc.add() == 7
        assert calc.multiply() == 12
        assert calc.sum_and_product == (7, 12)

    def test_dataclass_with_class_variables(self):
        """Test dataclass with class variables."""
        @dataclass
        class WithClassVars:
            instance_var: int
            class_var: int = field(init=False, default=100)

        data = dumps(WithClassVars)
        restored = loads(data)

        obj = restored(42)
        assert obj.instance_var == 42
        assert obj.class_var == 100

    def test_dataclass_inheritance(self):
        """Test dataclass inheritance."""
        @dataclass
        class Base:
            x: int

        @dataclass
        class Derived(Base):
            y: int

        data = dumps(Derived)
        restored = loads(data)

        obj = restored(10, 20)
        assert obj.x == 10
        assert obj.y == 20

    def test_dataclass_with_complex_defaults(self):
        """Test dataclass with complex default values."""
        def default_processor(x):
            return x * 2

        @dataclass
        class ComplexDefaults:
            name: str
            processor: callable = field(default=default_processor)
            config: Dict[str, int] = field(default_factory=lambda: {"a": 1, "b": 2})

        data = dumps(ComplexDefaults)
        restored = loads(data)

        obj = restored("test")
        assert obj.name == "test"
        assert obj.processor(5) == 10
        assert obj.config == {"a": 1, "b": 2}


class TestDataclassInstanceSerialization:
    """Tests for dataclass instance serialization."""

    def test_simple_instance(self):
        """Test serializing a dataclass instance."""
        @dataclass
        class Point:
            x: int
            y: int

        p = Point(10, 20)
        data = dumps(p)
        restored = loads(data)

        assert restored.x == 10
        assert restored.y == 20
        assert type(restored).__name__ == "Point"

    def test_instance_with_default_factory(self):
        """Test serializing instance with default_factory values."""
        @dataclass
        class Container:
            items: List[int] = field(default_factory=list)

        c = Container()
        c.items.extend([1, 2, 3])

        data = dumps(c)
        restored = loads(data)

        assert restored.items == [1, 2, 3]

    def test_frozen_instance(self):
        """Test serializing frozen dataclass instance."""
        @dataclass(frozen=True)
        class FrozenPoint:
            x: int
            y: int

        p = FrozenPoint(5, 10)
        data = dumps(p)
        restored = loads(data)

        assert restored.x == 5
        assert restored.y == 10

        with pytest.raises(AttributeError):
            restored.x = 100

    def test_nested_instance(self):
        """Test serializing nested dataclass instances."""
        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            inner: Inner
            name: str

        inner = Inner(100)
        outer = Outer(inner, "test")

        data = dumps(outer)
        restored = loads(data)

        assert restored.name == "test"
        assert restored.inner.value == 100
        assert type(restored.inner).__name__ == "Inner"

    def test_instance_with_methods(self):
        """Test that instance methods work after deserialization."""
        @dataclass
        class Rectangle:
            width: int
            height: int

            def area(self) -> int:
                return self.width * self.height

            def perimeter(self) -> int:
                return 2 * (self.width + self.height)

        rect = Rectangle(10, 5)
        data = dumps(rect)
        restored = loads(data)

        assert restored.area() == 50
        assert restored.perimeter() == 30

    def test_list_of_instances(self):
        """Test serializing a list of dataclass instances."""
        @dataclass
        class Item:
            id: int
            name: str

        items = [Item(1, "first"), Item(2, "second"), Item(3, "third")]

        data = dumps(items)
        restored = loads(data)

        assert len(restored) == 3
        assert restored[0].id == 1
        assert restored[1].name == "second"
        assert restored[2].id == 3

    def test_dict_with_instance_values(self):
        """Test serializing a dict with dataclass instance values."""
        @dataclass
        class Config:
            value: int
            enabled: bool

        configs = {
            "prod": Config(100, True),
            "dev": Config(1, False),
        }

        data = dumps(configs)
        restored = loads(data)

        assert restored["prod"].value == 100
        assert restored["prod"].enabled is True
        assert restored["dev"].value == 1
        assert restored["dev"].enabled is False


class TestDataclassEdgeCases:
    """Tests for edge cases in dataclass serialization."""

    def test_empty_dataclass(self):
        """Test dataclass with no fields."""
        @dataclass
        class Empty:
            pass

        data = dumps(Empty)
        restored = loads(data)

        obj = restored()
        assert obj is not None

    def test_dataclass_with_post_init(self):
        """Test dataclass with __post_init__."""
        @dataclass
        class WithPostInit:
            x: int
            y: int
            total: int = field(init=False)

            def __post_init__(self):
                self.total = self.x + self.y

        data = dumps(WithPostInit)
        restored = loads(data)

        obj = restored(10, 20)
        assert obj.total == 30

    def test_dataclass_with_init_false(self):
        """Test dataclass with init=False."""
        @dataclass(init=False)
        class NoInit:
            x: int
            y: int

            def __init__(self, value: int):
                self.x = value
                self.y = value * 2

        data = dumps(NoInit)
        restored = loads(data)

        obj = restored(5)
        assert obj.x == 5
        assert obj.y == 10

    def test_dataclass_with_repr_false(self):
        """Test dataclass with repr=False."""
        @dataclass(repr=False)
        class NoRepr:
            secret: str

        data = dumps(NoRepr)
        restored = loads(data)

        obj = restored("password123")
        assert obj.secret == "password123"
        # Should use default object repr, not dataclass repr
        assert "secret=" not in repr(obj)

    def test_deeply_nested_dataclasses(self):
        """Test deeply nested dataclass structures."""
        @dataclass
        class Level3:
            value: int

        @dataclass
        class Level2:
            level3: Level3

        @dataclass
        class Level1:
            level2: Level2

        @dataclass
        class Root:
            level1: Level1
            name: str

        # Create deeply nested structure
        l3 = Level3(42)
        l2 = Level2(l3)
        l1 = Level1(l2)
        root = Root(l1, "root")

        data = dumps(root)
        restored = loads(data)

        assert restored.name == "root"
        assert restored.level1.level2.level3.value == 42

    def test_dataclass_with_lambda_default(self):
        """Test dataclass with lambda in default_factory."""
        @dataclass
        class WithLambda:
            items: List[int] = field(default_factory=lambda: [1, 2, 3])
            transform: callable = field(default=lambda x: x * 2)

        data = dumps(WithLambda)
        restored = loads(data)

        obj = restored()
        assert obj.items == [1, 2, 3]
        assert obj.transform(5) == 10

    def test_dataclass_self_reference_through_optional(self):
        """Test dataclass with self-referential type through Optional."""
        @dataclass
        class TreeNode:
            value: int
            left: Optional["TreeNode"] = None
            right: Optional["TreeNode"] = None

        # Create a small tree
        leaf1 = TreeNode(1)
        leaf2 = TreeNode(3)
        root = TreeNode(2, leaf1, leaf2)

        data = dumps(root)
        restored = loads(data)

        assert restored.value == 2
        assert restored.left.value == 1
        assert restored.right.value == 3
        assert restored.left.left is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
