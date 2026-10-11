import os
import numpy as np
import pytest

import pylcp as fork

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_constant_magnetic_field_comparison():
    """Compare constant magnetic field evaluation against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "fields_ref.npz"))
    B0 = np.array([1.2, -0.5, 3.4])
    mag_fk = fork.ConstantMagneticField(B0)

    R = np.array([0.5, -1.0, 2.0])
    np.testing.assert_allclose(mag_fk.Field(R, 0), ref["const_field"])
    np.testing.assert_allclose(mag_fk.magnitude(R, 0), ref["const_mag"])
    np.testing.assert_allclose(mag_fk.gradient(R, 0), ref["const_grad"])


def test_quadrupole_magnetic_field_comparison():
    """Compare quadrupole magnetic field evaluation and gradient against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "fields_ref.npz"))
    grad = 15.0
    mag_fk = fork.QuadrupoleMagneticField(grad)

    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, -2.0, 3.0]),
        np.array([-0.5, 0.5, -1.0])
    ]
    for idx, R in enumerate(positions):
        np.testing.assert_allclose(mag_fk.Field(R, 0), ref["quad_fields"][idx])
        np.testing.assert_allclose(mag_fk.magnitude_gradient(R, 0), ref["quad_grad_mags"][idx])


def test_3d_mot_beams_comparison():
    """Compare 3D MOT 6-beam properties against precomputed upstream benchmark."""
    ref = np.load(os.path.join(DATA_DIR, "fields_ref.npz"))
    beams_fk = fork.Conventional3DMOTBeams(s=2.0, delta=-1.5)

    assert len(beams_fk.beam_vector) == len(ref["mot_deltas"])

    R0 = np.zeros(3)
    for idx, b_fk in enumerate(beams_fk.beam_vector):
        assert b_fk.delta(0) == pytest.approx(ref["mot_deltas"][idx])
        np.testing.assert_allclose(b_fk.kvec(R0, 0), ref["mot_kvecs"][idx])
