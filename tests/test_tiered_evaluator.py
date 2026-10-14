import numpy as np
import pytest
from pylcp.typing import Dependence, FieldParameter, ScalarFieldParameter, VectorFieldParameter
import pylcp as fork


def test_dependence_ordering():
    """Verify total ordering of Dependence Enum."""
    assert Dependence.CONSTANT < Dependence.POSITION_ONLY
    assert Dependence.POSITION_ONLY < Dependence.TIME_ONLY
    assert Dependence.TIME_ONLY < Dependence.POSITION_AND_TIME
    assert max([Dependence.CONSTANT, Dependence.POSITION_ONLY]) == Dependence.POSITION_ONLY
    assert max([Dependence.TIME_ONLY, Dependence.POSITION_ONLY]) == Dependence.TIME_ONLY


def test_field_parameter_constant():
    """Verify FieldParameter wraps constants directly without lambdas."""
    param = ScalarFieldParameter(5.0, "test_const")
    assert param.dependence == Dependence.CONSTANT
    assert param.val == 5.0
    assert param(R=np.array([1, 2, 3]), t=10.0) == 5.0


def test_field_parameter_spatial():
    """Verify FieldParameter detects spatial-only function signature."""
    def spatial_func(pos):
        return pos[0] * 2.0

    param = ScalarFieldParameter(spatial_func, "test_spatial")
    assert param.dependence == Dependence.POSITION_ONLY
    res = param(R=np.array([3.0, 0.0, 0.0]), t=5.0)
    assert res == 6.0


def test_field_parameter_temporal():
    """Verify FieldParameter detects temporal-only function signature."""
    def temporal_func(t):
        return t ** 2

    param = ScalarFieldParameter(temporal_func, "test_temporal")
    assert param.dependence == Dependence.TIME_ONLY
    res = param(R=np.array([0.0, 0.0, 0.0]), t=4.0)
    assert res == 16.0


def test_field_parameter_spatiotemporal():
    """Verify FieldParameter detects spatio-temporal function signature."""
    def full_func(pos, time):
        return pos[0] + time

    param = ScalarFieldParameter(full_func, "test_full")
    assert param.dependence == Dependence.POSITION_AND_TIME
    res = param(R=np.array([2.0, 0.0, 0.0]), t=3.0)
    assert res == 5.0


def test_governingeq_evaluator_strategy_selection():
    """Verify GoverningEq selects StaticEvaluator for constant/spatial setups."""
    H0_g, muq_g = fork.hamiltonians.singleF(F=0, gF=1.0)
    H0_e, muq_e = fork.hamiltonians.singleF(F=1, gF=1.0)
    dqij = fork.hamiltonians.dqij_two_bare_hyperfine(F=0, Fp=1, normalize=True)
    H = fork.Hamiltonian(H0_g, H0_e, muq_g, muq_e, dqij, mass=87.0)
    beams = fork.Conventional3DMOTBeams(s=1.0, delta=-1.0)
    mag = fork.ConstantMagneticField(np.array([0.0, 0.0, 0.0]))
    
    obe = fork.OBE(beams, mag, H, transform_into_re_im=False)
    
    assert obe.dependence == Dependence.CONSTANT
    assert isinstance(obe.evaluator, fork.governingeq.StaticEvaluator)


def test_scalar_field_parameter_validation():
    """Verify ScalarFieldParameter validation and type casting."""
    # Int converted to float
    p_int = ScalarFieldParameter(10, "int_param")
    assert isinstance(p_int(), float)
    assert p_int() == 10.0

    # Complex number supported
    p_cplx = ScalarFieldParameter(1.0 + 2.0j, "cplx_param")
    assert p_cplx() == 1.0 + 2.0j

    # Booleans rejected
    with pytest.raises(TypeError):
        ScalarFieldParameter(True, "bool_param")

    # Vectors rejected
    with pytest.raises(TypeError):
        ScalarFieldParameter([1.0, 2.0, 3.0], "vec_param")

    # Time-only callable called with positional argument
    p_t = ScalarFieldParameter(lambda t: 3.0 * t, "time_param")
    assert p_t(2.0) == 6.0

    # Callable returning int rejected
    with pytest.raises(TypeError):
        ScalarFieldParameter(lambda t: 1, "int_callable")


def test_vector_field_parameter_validation():
    """Verify VectorFieldParameter validation, shape checking, and dtype handling."""
    # List of ints converted to float ndarray of shape (3,)
    p_int_vec = VectorFieldParameter([1, 2, 3], "int_vec")
    res = p_int_vec()
    assert isinstance(res, np.ndarray)
    assert res.shape == (3,)
    assert res.dtype == np.float64
    np.testing.assert_array_equal(res, [1.0, 2.0, 3.0])

    # Complex vectors preserve complex dtype
    p_cplx_vec = VectorFieldParameter(np.array([1j, 0.0, 0.0]), "cplx_vec")
    assert np.issubdtype(p_cplx_vec().dtype, np.complexfloating)

    # Rejects scalars
    with pytest.raises(TypeError):
        VectorFieldParameter(5.0, "scalar_vec")

    # Rejects wrong shape (2 components or 4 components)
    with pytest.raises(ValueError):
        VectorFieldParameter([1.0, 2.0], "two_vec")
    with pytest.raises(ValueError):
        VectorFieldParameter([1.0, 2.0, 3.0, 4.0], "four_vec")

    # Callable returning vector
    p_func = VectorFieldParameter(lambda r: np.array([r[0], 0.0, 0.0]), "func_vec")
    assert p_func.dependence == Dependence.POSITION_ONLY
    out = p_func(np.array([5.0, 1.0, 2.0]))
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, [5.0, 0.0, 0.0])

    # Callable returning wrong shape rejected
    with pytest.raises(ValueError):
        VectorFieldParameter(lambda r: np.array([r[0], 0.0]), "bad_func")

    # Callable returning list rejected
    with pytest.raises(TypeError):
        VectorFieldParameter(lambda r: [r[0], 0.0, 0.0], "list_func")

    # Callable returning int array rejected
    with pytest.raises(TypeError):
        VectorFieldParameter(lambda r: np.array([1, 2, 3]), "int_arr_func")


def test_beam_parameter_types_and_dependence():
    """Verify LaserBeam field parameter return types and dependency propagation."""
    beam = fork.LaserBeam(
        kvec=[0, 0, 1],
        s=2.5,
        pol=fork.PolarizationEnum.SIGMA_PLUS,
        delta=-1.0
    )
    # Return types
    assert isinstance(beam.intensity(), float)
    assert beam.intensity() == 2.5
    assert isinstance(beam.kvec(), np.ndarray)
    assert beam.kvec().shape == (3,)
    assert isinstance(beam.pol(), np.ndarray)
    assert beam.pol().shape == (3,)
    assert np.issubdtype(beam.pol().dtype, np.complexfloating)

    # Dependence
    assert beam.dependence == Dependence.CONSTANT

    # Time-dependent beam
    t_beam = fork.LaserBeam(
        kvec=[0, 0, 1],
        s=lambda t: 2.5 * t,
        pol=fork.PolarizationEnum.PI,
        delta=0.0
    )
    assert t_beam.dependence == Dependence.TIME_ONLY
    assert t_beam.intensity(t=2.0) == 5.0

    # Gaussian beam spatial dependence
    g_beam = fork.GaussianBeam(
        kvec=np.array([0., 0., 1.]),
        pol=fork.PolarizationEnum.SIGMA_PLUS,
        s=1.0,
        delta=0.0,
        wb=1.0
    )
    assert g_beam.dependence == Dependence.POSITION_ONLY

