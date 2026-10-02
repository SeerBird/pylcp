import numpy as np
import pytest
import tests.vendor.upstream_pylcp as upstream
import pylcp as fork

def test_constant_magnetic_field_comparison():
    """Compare constant magnetic field evaluation between upstream and fork."""
    B0 = np.array([1.2, -0.5, 3.4])
    mag_up = upstream.constantMagneticField(B0)
    mag_fk = fork.ConstantMagneticField(B0)
    
    R = np.array([0.5, -1.0, 2.0])
    np.testing.assert_allclose(mag_up.Field(R, 0), mag_fk.Field(R, 0))
    np.testing.assert_allclose(mag_up.FieldMag(R, 0), mag_fk.magnitude(R, 0))
    np.testing.assert_allclose(mag_up.gradField(R, 0), mag_fk.gradient(R, 0))

def test_quadrupole_magnetic_field_comparison():
    """Compare quadrupole magnetic field evaluation and gradient between upstream and fork."""
    grad = 15.0
    mag_up = upstream.quadrupoleMagneticField(grad)
    mag_fk = fork.QuadrupoleMagneticField(grad)
    
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, -2.0, 3.0]),
        np.array([-0.5, 0.5, -1.0])
    ]
    for R in positions:
        np.testing.assert_allclose(mag_up.Field(R, 0), mag_fk.Field(R, 0))
        np.testing.assert_allclose(mag_up.gradFieldMag(R, 0), mag_fk.magnitude_gradient(R, 0))

def test_3d_mot_beams_comparison():
    """Compare 3D MOT 6-beam properties between upstream and fork."""
    beams_up = upstream.conventional3DMOTBeams(s=2.0, delta=-1.5)
    beams_fk = fork.Conventional3DMOTBeams(s=2.0, delta=-1.5)
    
    assert len(beams_up.beam_vector) == len(beams_fk.beam_vector)
    
    R0 = np.zeros(3)
    for b_up, b_fk in zip(beams_up.beam_vector, beams_fk.beam_vector):
        assert b_up.delta(0) == pytest.approx(b_fk.delta(0))
        np.testing.assert_allclose(b_up.kvec(R0, 0), b_fk.kvec(R0, 0))
