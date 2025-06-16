import torch
import inspect
from nnsight import autoamp_init_wrapper


def test_autoamp_init_wrapper_for_autocast():
    # Get a reference to the original init and examine its signature
    original_autocast_init = torch.autocast.__init__

    # Print the original signature for reference
    sig = inspect.signature(original_autocast_init)
    print(f"Original torch.autocast.__init__ signature: {sig}")

    original_calls = []

    # Part 1: Check if the Wrapper passes the arguments correctly
    try:
        # Create a mock function that will record how it was called
        def mock_init(self, device_type, **kwargs):
            original_calls.append((device_type, kwargs))
            return None

        # Apply our wrapper to the mock function
        wrapped_mock_init = autoamp_init_wrapper(mock_init)

        # Test case 1: Basic usage with positional and keyword args
        wrapped_mock_init(None, "cpu", dtype=torch.bfloat16)
        assert original_calls[-1] == ("cpu", {"dtype": torch.bfloat16})

        # Test case 2: With more than one keyword argument
        wrapped_mock_init(None, "cuda", dtype=torch.float16, enabled=False)
        assert original_calls[-1] == ("cuda", {"dtype": torch.float16, "enabled": False})

        # Test case 3: With meta device (which triggers the special handling)
        wrapped_mock_init(None, "meta")
        assert original_calls[-1] == ("meta", {
            "dtype": torch.get_autocast_dtype('cpu')
        })

        # Test case 4: With CUDA device and other kwargs
        wrapped_mock_init(None, "cuda", enabled=True, cache_enabled=False)
        assert original_calls[-1] == ("cuda", {
            "enabled": True,
            "cache_enabled": False
        })
    finally:
        original_calls.clear()  # Clear the calls for the next part

    # Part 2: Test that the Wrapper can be applied to torch.autocast.__init__
    try:
        # NOTE: The 'meta' device is not supported in the current PyTorch version, so we will not test it here.
        # Throws the following error if we try to use it.
        # RuntimeError: User specified an unsupported autocast device_type 'meta'

        # Create a tracker to see how torch.autocast.__init__ is actually called
        def original_autocast_init_with_tracking(self, device_type, **kwargs):
            original_calls.append((device_type, kwargs))
            return original_autocast_init(self, device_type, **kwargs)

        # Monkey-patch the original __init__ method (with tracking) with our wrapper
        torch.autocast.__init__ = autoamp_init_wrapper(original_autocast_init_with_tracking)

        # Test all parameter combinations based on the signature:
        # device_type: str, dtype: Optional[_dtype] = None, enabled: bool = True, cache_enabled: Optional[bool] = None

        # Test Case 2.1: Default parameters (only device_type)
        autocast_instance = torch.autocast("cpu")

        # Verify the last call matches expected values
        assert original_calls[-1][0] == "cpu"
        assert "dtype" not in original_calls[-1][1]
        assert "enabled" not in original_calls[-1][1]
        assert "cache_enabled" not in original_calls[-1][1]

        # Verify the instance properties match expected values
        assert autocast_instance.device == "cpu"
        assert autocast_instance.fast_dtype == torch.get_autocast_dtype("cpu")  # Default dtype for CPU
        assert autocast_instance._enabled == True  # Default is True
        assert autocast_instance._cache_enabled is torch.is_autocast_cache_enabled()  # Default

        # Test Case 2.2: With explicit dtype
        autocast_instance = torch.autocast("cpu", dtype=torch.bfloat16)

        # Verify the last call matches expected values
        assert original_calls[-1][0] == "cpu"
        assert original_calls[-1][1]["dtype"] == torch.bfloat16
        assert "enabled" not in original_calls[-1][1]
        assert "cache_enabled" not in original_calls[-1][1]

        # Verify the instance properties match expected values
        assert autocast_instance.device == "cpu"
        assert autocast_instance.fast_dtype == torch.bfloat16
        assert autocast_instance._enabled == True  # Default is True
        assert autocast_instance._cache_enabled is torch.is_autocast_cache_enabled()  # Default

        # Test Case 2.3: With explicit enabled=False
        autocast_instance = torch.autocast("cpu", enabled=False)

        # Verify the last call matches expected values
        assert original_calls[-1][0] == "cpu"
        assert "dtype" not in original_calls[-1][1]
        assert original_calls[-1][1]["enabled"] == False
        assert "cache_enabled" not in original_calls[-1][1]

        # Verify the instance properties match expected values
        assert autocast_instance.device == "cpu"
        assert autocast_instance.fast_dtype == torch.get_autocast_dtype("cpu")  # Default dtype for CPU
        assert autocast_instance._enabled == False
        assert autocast_instance._cache_enabled is torch.is_autocast_cache_enabled()  # Default

        # Test Case 2.4: With explicit cache_enabled
        autocast_instance = torch.autocast("cpu", cache_enabled=False)

        # Verify the last call matches expected values
        assert original_calls[-1][0] == "cpu"
        assert "dtype" not in original_calls[-1][1]
        assert "enabled" not in original_calls[-1][1]
        assert original_calls[-1][1]["cache_enabled"] == False

        # Verify the instance properties match expected values
        assert autocast_instance.device == "cpu"
        assert autocast_instance.fast_dtype == torch.get_autocast_dtype("cpu")  # Default dtype for CPU
        assert autocast_instance._enabled == True  # Default is True
        assert autocast_instance._cache_enabled == False

        # Test Case 2.5: With all parameters specified
        autocast_instance = torch.autocast("cpu", dtype=torch.bfloat16,
                                           enabled=False, cache_enabled=False)

        # Verify the last call matches expected values
        assert original_calls[-1][0] == "cpu"
        assert original_calls[-1][1]["dtype"] == torch.bfloat16
        assert original_calls[-1][1]["enabled"] == False
        assert original_calls[-1][1]["cache_enabled"] == False

        # Verify the instance properties match expected values
        assert autocast_instance.device == "cpu"
        assert autocast_instance.fast_dtype == torch.bfloat16
        assert autocast_instance._enabled == False
        assert autocast_instance._cache_enabled == False

        # Print summary of tested parameter combinations
        print(f"All parameter combinations tested with wrapper applied: {original_calls}")

    finally:
        torch.autocast.__init__ = original_autocast_init  # Restore original function
        original_calls.clear()  # Clear the calls for the next part
