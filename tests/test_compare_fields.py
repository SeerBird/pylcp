import os
import numpy as np
import pytest

import pylcp as fork
from tests.test_configs import (
    CONST_B_VECTOR, CONST_B_POS,
    QUAD_B_GRADIENT, QUAD_B_POSITIONS,
    MOT_BEAMS_CONFIG
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_constant_magnetic_field_comparison():
    """Compare constant magnetic field evaluation against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "fields_ref.npz"))
    mag_fk = fork.ConstantMagneticField(CONST_B_VECTOR)

    np.testing.assert_allclose(mag_fk.Field(CONST_B_POS, 0), ref["const_field"])
    np.testing.assert_allclose(mag_fk.magnitude(CONST_B_POS, 0), ref["const_mag"])
    np.testing.assert_allclose(mag_fk.gradient(CONST_B_POS, 0), ref["const_grad"])


def test_quadrupole_magnetic_field_comparison():
    """Compare quadrupole magnetic field evaluation and gradient against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "fields_ref.npz"))
    mag_fk = fork.QuadrupoleMagneticField(QUAD_B_GRADIENT)

    for idx, R in enumerate(QUAD_B_POSITIONS):
        np.testing.assert_allclose(mag_fk.Field(R, 0), ref["quad_fields"][idx])
        np.testing.assert_allclose(mag_fk.magnitude_gradient(R, 0), ref["quad_grad_mags"][idx])


def test_3d_mot_beams_comparison():
    """Compare 3D MOT 6-beam properties against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "fields_ref.npz"))
    beams_fk = fork.Conventional3DMOTBeams(**MOT_BEAMS_CONFIG)

    assert len(beams_fk.beam_vector) == len(ref["mot_deltas"])

    R0 = np.zeros(3)
    for idx, b_fk in enumerate(beams_fk.beam_vector):
        assert b_fk.delta(0) == pytest.approx(ref["mot_deltas"][idx])
        np.testing.assert_allclose(b_fk.kvec(R0, 0), ref["mot_kvecs"][idx])
