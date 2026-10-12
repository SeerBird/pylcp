from typing import overload

import numpy as np

from pylcp_gui.util import HFTransitionKey, Polarization, PolarizationEnum, FineTransitionKey, HyperfineKey


class StateData:
    label: str
    energy: float
    I: float
    J: float
    hf_coefs: tuple[float, float, float]
    gJ: float
    # list of lists of mF values for each possible F each mF list is sorted in increasing mF order
    substates: dict[float, list[float]]

    @overload
    def __init__(self, other: StateData, /):
        """Initialize a StateData instance by copying from an existing StateData object."""

    @overload
    def __init__(self, label: str, energy: float, I: float, J: float,
                 hf_coefs: tuple[float, float, float], gJ: float):
        """Initialize a StateData instance from explicit state parameters."""

    def __init__(self, label: str | StateData,
                 energy: float | None = None,
                 I: float | None = None,
                 J: float | None = None,
                 hf_coefs: tuple[float, float, float] | None = None,
                 gJ: float | None = None):
        if isinstance(label, StateData):
            other = label
            self.label = other.label
            self.energy = other.energy
            self.I = other.I
            self.J = other.J
            self.hf_coefs = other.hf_coefs
            self.gJ = other.gJ
            self.substates = {F: list(mFs) for F, mFs in other.substates.items()}
        else:
            if energy is None or I is None or J is None or hf_coefs is None or gJ is None:
                raise ValueError("energy, I, J, hf_coefs, and gJ must be provided when "
                                 + "constructing StateData from individual arguments.")
            self.label = str(label)
            self.energy = float(energy)
            self.I = float(I)
            self.J = float(J)
            self.hf_coefs = hf_coefs
            self.gJ = float(gJ)
            self.substates = {}
            Fs = np.arange(np.abs(J - I), J + I + 1, 1)
            for F in Fs:
                self.substates[F] = list(np.arange(-F, F + 1, 1.))
    def hyperfine_keys(self):
        return [HyperfineKey(self.label, F) for F in self.substates]

class TransitionData:
    def __init__(self, gamma):
        self.gamma = gamma  # Hz


class LaserTransitionGroup[T:LaserFreqGroup]:
    transition: FineTransitionKey
    freq_groups: dict[float, T]

    @overload
    def __init__(self, other: LaserTransitionGroup, /):
        """Initialize a LaserTransitionGroup instance by copying from an existing LaserTransitionGroup object."""

    @overload
    def __init__(self, transition: FineTransitionKey):
        """Initialize a LaserTransitionGroup instance from explicit transition key."""

    def __init__(self, transition: FineTransitionKey | LaserTransitionGroup):
        if isinstance(transition, LaserTransitionGroup):
            other = transition
            self.transition = other.transition
            self.freq_groups = {freq: LaserFreqGroup(fg) for freq, fg in other.freq_groups.items()}
        else:
            self.transition = transition
            self.freq_groups = {}

    def freqs(self):
        return self.freq_groups.keys()

    def __iter__(self):
        yield from self.freq_groups.values()


class LaserFreqGroup[T:LaserData]:
    freq: float
    transition: FineTransitionKey
    lasers: list[T]
    enabled_transitions: list[HFTransitionKey]

    @overload
    def __init__(self, other: LaserFreqGroup, /):
        """Initialize a LaserFreqGroup instance by copying from an existing LaserFreqGroup object."""

    @overload
    def __init__(self, freq: float, transition: FineTransitionKey,
                 enabled_transitions: list[HFTransitionKey] | None = None):
        """Initialize a LaserFreqGroup instance from explicit frequency and transition key,
        and optional enabled_transitions"""

    def __init__(self, freq: float | LaserFreqGroup, transition: FineTransitionKey | None = None,
                 enabled_transitions: list[HFTransitionKey] | None = None):
        if isinstance(freq, LaserFreqGroup):
            other = freq
            self.freq = float(other.freq)
            self.transition = other.transition
            self.lasers = [LaserData(laser) for laser in other.lasers]
            self.enabled_transitions = list(other.enabled_transitions)
        else:
            if transition is None:
                raise ValueError(
                    "transition must be provided when constructing LaserFreqGroup from raw arguments.")
            self.freq = float(freq)
            self.transition = transition
            self.lasers = []
            self.enabled_transitions = [] if enabled_transitions is None else enabled_transitions

    def add_laser(self, laser: T):
        self.lasers.append(laser)

    def __iter__(self):
        yield from self.lasers


class LaserData:
    freq: float
    kvec: np.ndarray
    pol: Polarization
    intensity: float

    @overload
    def __init__(self, other: LaserData, /):
        """Initialize a LaserData instance by copying from an existing LaserData object."""

    @overload
    def __init__(self, freq: float, kvec: np.ndarray, pol: Polarization, intensity: float):
        """Initialize a LaserData instance from explicit frequency, k-vector, polarization, and intensity."""

    def __init__(self, freq: float | LaserData,
                 kvec: np.ndarray | None = None,
                 pol: Polarization | None = None,
                 intensity: float | None = None):
        if isinstance(freq, LaserData):
            # Signature 1: Copy from existing LaserData
            other = freq
            self.freq = other.freq
            self.kvec = np.array(other.kvec, copy=True)
            self.pol = other.pol
            self.intensity = other.intensity
        else:
            # Signature 2: Initialize from separate arguments
            if kvec is None or pol is None or intensity is None:
                raise ValueError(
                    "kvec, pol, and intensity must be provided when constructing LaserData from individual arguments.")
            self.freq = float(freq)
            self.kvec = np.array(kvec, copy=True)
            self.pol = pol
            self.intensity = float(intensity)

    def __str__(self):
        return (f"kvec = ({self.kvec[0]},{self.kvec[1]},{self.kvec[2]}), " +
                f"pol = ({self.pol[0]},{self.pol[1]},{self.pol[2]})" +
                f"intensity = {self.intensity}")


class LaserDisplayData:
    def __init__(self, freq: float, keys: HFTransitionKey, orientation: bool):
        self.freq: float = freq
        self.keys: HFTransitionKey = keys
        self.upwards: bool = orientation
