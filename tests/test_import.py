import pylcp

def test_pylcp_import():
    """Verify pylcp can be imported successfully."""
    assert pylcp.__file__ is not None

def test_pylcp_version():
    """Verify pylcp has a version attribute."""
    assert hasattr(pylcp, "__version__")
    assert pylcp.__version__ == "1.0.2"

def test_basic_exports():
    """Verify modern PascalCase classes and types are exported at top-level."""
    assert hasattr(pylcp, "Atom")
    assert hasattr(pylcp, "State")
    assert hasattr(pylcp, "Transition")
    assert hasattr(pylcp, "Hamiltonian")
    assert hasattr(pylcp, "RateEq")
    assert hasattr(pylcp, "OBE")
    assert hasattr(pylcp, "HeuristicEq")
    assert hasattr(pylcp, "MagField")
    assert hasattr(pylcp, "LaserBeams")
    assert hasattr(pylcp, "TransitionKey")
    assert hasattr(pylcp, "Dependence")
