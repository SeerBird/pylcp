import pylcp

def test_pylcp_import():
    """Verify pylcp can be imported successfully."""
    assert pylcp.__file__ is not None

def test_pylcp_version():
    """Verify pylcp has a version attribute."""
    assert hasattr(pylcp, "__version__")
    assert pylcp.__version__ == "1.0.2"

def test_basic_exports():
    """Verify core classes are exported at top-level."""
    assert hasattr(pylcp, "atom")
    assert hasattr(pylcp, "hamiltonian")
    assert hasattr(pylcp, "rateeq")
    assert hasattr(pylcp, "obe")
