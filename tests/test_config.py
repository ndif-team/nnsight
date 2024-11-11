import pytest
from pathlib import Path

@pytest.mark.config
def test_pyproject_toml_exists():
    """Ensure that pyproject.toml file exists in the project root."""
    assert Path("pyproject.toml").is_file(), "pyproject.toml file is missing"

@pytest.mark.config
def test_pyproject_toml_validity(load_pyproject_toml):
    """Ensure pyproject.toml is correctly formatted and valid."""
    # If the fixture `load_pyproject_toml` runs successfully, the file is valid
    data = load_pyproject_toml
    assert data, "pyproject.toml is empty or could not be read"

    # Example check: Ensure required fields are present
    required_fields = ["project", "build-system"]
    for field in required_fields:
        assert field in data, f"Missing required field '{field}' in pyproject.toml"
