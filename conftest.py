import pytest
import toml

def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda:0")
    parser.addoption(
        "--tp", 
        action="store", 
        type=int, 
        default="1", 
        help="An argument for specifying the number of gpus to be used by VLLM"
    )

def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.device
    if "device" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("device", [option_value], scope="module")

@pytest.fixture(scope="session")
def load_pyproject_toml():
    """Fixture to load and parse the pyproject.toml file."""
    try:
        with open("pyproject.toml", "r") as f:
            data = toml.load(f)
        return data
    except toml.TomlDecodeError as e:
        pytest.fail(f"Failed to load pyproject.toml: {e}")

collect_ignore = ["examples/test_server.py", "examples/test_server_llama.py"]
