"""
author: SPE

Basics of the lcp physics package
"""
__version__ = "1.0.2"
import numpy as np

from . import hamiltonians
from .atom import atom
from .heuristiceq import heuristiceq
from .rateeq import rateeq
from .obe import obe
from .hamiltonian import hamiltonian
from .fields import (magField, constantMagneticField, quadrupoleMagneticField, iPMagneticField,
                     laserBeam, laserBeams, infinitePlaneWaveBeam, gaussianBeam,
                     clippedGaussianBeam, conventional3DMOTBeams)
