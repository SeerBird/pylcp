import numpy as np
import pytest
from pylcp.typing import Dependence, FieldParameter, ValidationType
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
    param = FieldParameter(5.0, "test_const",ValidationType.NumericScalar)
    assert param.dependence == Dependence.CONSTANT
    assert param.val == 5.0
    assert param(R=np.array([1, 2, 3]), t=10.0) == 5.0


def test_field_parameter_spatial():
    """Verify FieldParameter detects spatial-only function signature."""
    def spatial_func(pos):
        return pos[0] * 2.0

    param = FieldParameter(spatial_func, "test_spatial",ValidationType.NumericScalar)
    assert param.dependence == Dependence.POSITION_ONLY
    res = param(R=np.array([3.0, 0.0, 0.0]), t=5.0)
    assert res == 6.0


def test_field_parameter_temporal():
    """Verify FieldParameter detects temporal-only function signature."""
    def temporal_func(t):
        return t ** 2

    param = FieldParameter(temporal_func, "test_temporal",ValidationType.NumericScalar)
    assert param.dependence == Dependence.TIME_ONLY
    res = param(R=np.array([0.0, 0.0, 0.0]), t=4.0)
    assert res == 16.0


def test_field_parameter_spatiotemporal():
    """Verify FieldParameter detects spatio-temporal function signature."""
    def full_func(pos, time):
        return pos[0] + time

    param = FieldParameter(full_func, "test_full",ValidationType.NumericScalar)
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
