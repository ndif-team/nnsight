"""
Demo script to show exception output with and without DEBUG mode.

Run with: conda run -n ns312 python tests/debug_demo.py
"""

import torch
from collections import OrderedDict
from nnsight import NNsight, CONFIG

# Create tiny model
net = torch.nn.Sequential(
    OrderedDict(
        [
            ("layer1", torch.nn.Linear(5, 10)),
            ("layer2", torch.nn.Linear(10, 2)),
        ]
    )
)
model = NNsight(net)
inp = torch.rand((1, 5))


def demo_out_of_order():
    """Demonstrate OutOfOrderError."""
    print("=" * 70)
    print("OutOfOrderError Demo")
    print("=" * 70)

    print("\n--- DEBUG = False (default) ---\n")
    CONFIG.APP.DEBUG = False
    try:
        with model.trace(inp):
            out2 = model.layer2.output.save()
            out1 = model.layer1.output.save()  # Out of order!
    except Exception as e:
        print(str(e))

    print("\n--- DEBUG = True ---\n")
    CONFIG.APP.DEBUG = True
    try:
        with model.trace(inp):
            out2 = model.layer2.output.save()
            out1 = model.layer1.output.save()  # Out of order!
    except Exception as e:
        print(str(e))


def demo_index_error():
    """Demonstrate IndexError with bad tensor access."""
    print("\n" + "=" * 70)
    print("IndexError Demo")
    print("=" * 70)

    print("\n--- DEBUG = False ---\n")
    CONFIG.APP.DEBUG = False
    try:
        with model.trace(inp):
            bad = model.layer1.output[999].save()
    except Exception as e:
        print(str(e))

    print("\n--- DEBUG = True ---\n")
    CONFIG.APP.DEBUG = True
    try:
        with model.trace(inp):
            bad = model.layer1.output[999].save()
    except Exception as e:
        print(str(e))


def demo_helper_function():
    """Demonstrate error in helper function."""
    print("\n" + "=" * 70)
    print("Error in Helper Function Demo")
    print("=" * 70)

    def bad_helper(m):
        # Error happens here
        return m.layer1.output[999]

    print("\n--- DEBUG = False ---\n")
    CONFIG.APP.DEBUG = False
    try:
        with model.trace(inp):
            result = bad_helper(model).save()
    except Exception as e:
        print(str(e))

    print("\n--- DEBUG = True ---\n")
    CONFIG.APP.DEBUG = True
    try:
        with model.trace(inp):
            result = bad_helper(model).save()
    except Exception as e:
        print(str(e))


def demo_backward_error():
    """Demonstrate error in backward context."""
    print("\n" + "=" * 70)
    print("Error in Backward Context Demo")
    print("=" * 70)

    print("\n--- DEBUG = False ---\n")
    CONFIG.APP.DEBUG = False
    try:
        with model.trace(inp):
            out = model.layer1.output
            loss = model.output.sum()
            with loss.backward():
                bad_grad = out.grad[999].save()
    except Exception as e:
        print(str(e))

    print("\n--- DEBUG = True ---\n")
    CONFIG.APP.DEBUG = True
    try:
        with model.trace(inp):
            out = model.layer1.output
            loss = model.output.sum()
            with loss.backward():
                bad_grad = out.grad[999].save()
    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    demo_out_of_order()
    demo_index_error()
    demo_helper_function()
    demo_backward_error()

    # Reset DEBUG
    CONFIG.APP.DEBUG = False
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
