"""
author: SPE

Basics of the lcp physics package
"""
__version__ = "1.0.2"
import numpy as np

from . import hamiltonians
from .atom import State, Transition, Atom
from .heuristiceq import HeuristicEq
from .rateeq import RateEq
from .obe import OBE
from .hamiltonian import Hamiltonian
from .fields import (MagField, ConstantMagneticField, QuadrupoleMagneticField, IPMagneticField,
                     LaserBeam, LaserBeams, InfinitePlaneWaveBeam, GaussianBeam,
                     ClippedGaussianBeam, Conventional3DMOTBeams)
from .typing import Signature, TransitionKey, OBEEvolutionMatrices, RateEqEvolutionMatrices
