from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Union, Iterator, Literal, Callable, TYPE_CHECKING, NamedTuple, TypeAlias, \
    TypeVar, Generic
import numpy as np
import inspect

Vector3D = np.ndarray[tuple[Literal[3], ...], np.dtype[np.float64] | np.dtype[np.complex128]]


class Dependence(Enum):
    """Function argument signatures used for field promotion and parameter metadata."""
    CONSTANT = 0
    POSITION_ONLY = 1
    TIME_ONLY = 2
    POSITION_AND_TIME = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


def _inspect_callable_dependence(func: Callable) -> Dependence:
    """Get Dependence and validate signature (check if func accepts zeroes)"""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    if len(params) == 0:
        raise ValueError("Please pass a constant instead of a zero-parameter callable")
    elif len(params) == 1:
        parameter = params[0].lower()
        if parameter in ('t', 'time'):
            try:
                func(0.0)
            except Exception:
                raise ValueError(
                    "Seemingly time-dependent field parameter didn't accept 0 as a parameter")
            return Dependence.TIME_ONLY
        else:
            try:
                func(np.zeros(3))
            except Exception:
                raise ValueError(
                    "Seemingly space-dependent field parameter didn't accept" +
                    " np.zeros(3) as a parameter")
            return Dependence.POSITION_ONLY
    else:  # 2 params
        try:
            func(np.zeros(3), 0.0)
        except Exception:
            raise ValueError("A two-argument function should take position first and time second")
        return Dependence.POSITION_AND_TIME


# Scalar types:
NumericScalar = float | int | complex | np.number

# Vector / Array constant types:
VectorLike = np.ndarray | list[float] | tuple[float, ...] | list[complex] | tuple[complex, ...]

# Callable field functions: f(R), f(t), f(R, t) or general callable
FieldCallable = (Callable[[np.ndarray], VectorLike | NumericScalar]
                 | Callable[[float], VectorLike | NumericScalar]
                 | Callable[[np.ndarray, float], VectorLike | NumericScalar])

# Full comprehensive parameter value type:
FieldParameterValue = NumericScalar | VectorLike | FieldCallable

T = TypeVar("T")


class FieldParameter(Generic[T]):
    """
    Wraps a scalar, vector, or function parameter with explicit dependency metadata.
    """
    val: Any
    name: str
    dependence: Dependence

    def __init__(self, val: Any, name: str = ""):
        self.name = name
        if not callable(val):
            self.dependence = Dependence.CONSTANT
            self.val = self._validate_constant(val)
        else:
            self.val = val
            self.dependence = _inspect_callable_dependence(val)
            self._validate_callable(val)

    def _validate_constant(self, val: Any) -> Any:
        return val

    def _validate_callable(self, func: Callable) -> None:
        pass

    def __call__(self, R: np.ndarray | None = None, t: float = 0.) -> T:
        if self.dependence == Dependence.CONSTANT:
            return self.val

        if self.dependence == Dependence.TIME_ONLY:
            time_val = t if R is None else (R if isinstance(R, (int, float, np.number)) else t)
            return self.val(time_val)

        pos = np.zeros(3) if R is None else R
        if self.dependence == Dependence.POSITION_ONLY:
            return self.val(pos)
        else:
            return self.val(pos, t)


class ScalarFieldParameter(FieldParameter[float]):
    """
    Field parameter returning a scalar float or complex number.
    """
    def _validate_constant(self, val: Any) -> float | complex:
        if isinstance(val, bool):
            raise TypeError(f"{self.name.capitalize()} must be a float or complex number, got bool")
        if isinstance(val, (int, float, np.floating, np.integer)):
            return float(val)
        elif isinstance(val, (complex, np.complexfloating)):
            return complex(val)
        else:
            raise TypeError(f"{self.name.capitalize()} must be a float or complex number, got {type(val).__name__}")

    def _validate_callable(self, func: Callable) -> None:
        res = self(np.zeros(3), 0.0)
        if isinstance(res, bool) or not isinstance(res, (float, complex, np.floating, np.complexfloating)):
            raise TypeError(f"{self.name.capitalize()} callable must return a float or complex number, got {type(res).__name__}")


class VectorFieldParameter(FieldParameter[np.ndarray]):
    """
    Field parameter returning a 3-element vector (numpy ndarray).
    """
    def _validate_constant(self, val: Any) -> np.ndarray:
        if not isinstance(val, (list, tuple, np.ndarray)):
            raise TypeError(f"{self.name.capitalize()} must be an array-like of 3 float or complex numbers, got {type(val).__name__}")
        if len(val) != 3:
            raise ValueError(f"{self.name.capitalize()} must be a vector with 3 components, got {len(val)}")
        arr = np.asarray(val)
        if arr.shape != (3,):
            raise ValueError(f"{self.name.capitalize()} must be a vector with 3 components, got {arr.shape}")
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(float)
        elif not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating)):
            raise TypeError(f"{self.name.capitalize()} must be an array-like of 3 float or complex numbers, got {arr.dtype}")
        return arr

    def _validate_callable(self, func: Callable) -> None:
        res = self(np.zeros(3), 0.0)
        if not isinstance(res, np.ndarray):
            raise TypeError(f"{self.name.capitalize()} callable must return a numpy ndarray, got {type(res).__name__}")
        if res.shape != (3,):
            raise ValueError(f"{self.name.capitalize()} callable must return a vector with shape (3,), got {res.shape}")
        if np.issubdtype(res.dtype, np.integer):
            raise TypeError(f"{self.name.capitalize()} callable must return a float or complex array, got integer dtype {res.dtype}")


class CoordBasis(Enum):
    SPHERICAL = auto()
    CARTESIAN = auto()


class PolarizationEnum(Enum):
    """Enumeration of standard laser polarization states relative to beam k-vector."""
    SIGMA_PLUS = 1
    PI = 0
    SIGMA_MINUS = -1

    def to_array(self)->np.ndarray:
        """relative to the kvector"""
        if self.value == PolarizationEnum.SIGMA_PLUS.value:
            return np.array([0., 0., 1.], dtype='complex')
        elif self.value == PolarizationEnum.PI.value:
            return np.array([0., 1., 0.], dtype='complex')
        else:
            return np.array([1., 0., 0.], dtype='complex')

    def opposite(self):
        if self.value == PolarizationEnum.SIGMA_PLUS.value:
            return PolarizationEnum.SIGMA_MINUS
        elif self.value == PolarizationEnum.PI.value:
            return PolarizationEnum.PI
        else:
            return PolarizationEnum.SIGMA_PLUS


Polarization: TypeAlias = (Vector3D
                           | PolarizationEnum
                           | Callable[[Vector3D, float], Vector3D]
                           | Callable[[Vector3D], Vector3D])


class HyperfineKey(NamedTuple):
    label: str
    F: float

    def __str__(self) -> str:
        return f"{self.label}_F{self.F:g}"


class MagneticKey(NamedTuple):
    label: str
    F: float
    mF: float


class TransitionKey(NamedTuple):
    """Strongly typed representation of an atomic transition key (e.g. g->e)."""
    lower: str
    upper: str

    def __str__(self) -> str:
        return f"{self.lower}->{self.upper}"


class HFTransitionKey(NamedTuple):
    lower_key: HyperfineKey
    upper_key: HyperfineKey

    def __str__(self) -> str:
        return f"{self.lower_key}->{self.upper_key}"

    def to_transition(self) -> TransitionKey:
        return TransitionKey(self.lower_key.label, self.upper_key.label)


# region evolution matrix containers
@dataclass
class OBEEvolutionMatrices:
    """
    Structured container for Optical Bloch Equation evolution matrices.
    
    Provides both attribute access (.decay, .h0, .magnetic, .re_electric, .im_electric, .d_q, .d_q_conj)
    and dictionary-compatible indexing (['decay'], ['H0'], ['B'], ['reE'], ['imE'], ['d_q'], ['d_q*']).
    """
    decay: np.ndarray = field(default_factory=lambda: np.array([]))
    h0: np.ndarray = field(default_factory=lambda: np.array([]))
    static: np.ndarray = field(default_factory=lambda: np.array([]))
    re_electric: dict[TransitionKey, Any] = field(default_factory=dict)
    im_electric: dict[TransitionKey, Any] = field(default_factory=dict)
    d_q: dict[TransitionKey, Any] = field(default_factory=dict)
    d_q_conj: dict[TransitionKey, Any] = field(default_factory=dict)
    magnetic: Any = field(default_factory=list)

    def __getitem__(self, key: Union[str, TransitionKey]) -> Any:
        mapping = {
            "decay": self.decay,
            "H0": self.h0,
            "h0": self.h0,
            "static": self.static,
            "B": self.magnetic,
            "magnetic": self.magnetic,
            "reE": self.re_electric,
            "re_electric": self.re_electric,
            "imE": self.im_electric,
            "im_electric": self.im_electric,
            "d_q": self.d_q,
            "d_q*": self.d_q_conj,
            "d_q_conj": self.d_q_conj,
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(f"Invalid key '{key}' for OBEEvolutionMatrices")

    def __setitem__(self, key: Union[str, TransitionKey], value: Any) -> None:
        if key == "decay":
            self.decay = value
        elif key in ("H0", "h0"):
            self.h0 = value
        elif key == "static":
            self.static = value
        elif key in ("B", "magnetic"):
            self.magnetic = value
        elif key in ("reE", "re_electric"):
            self.re_electric = value
        elif key in ("imE", "im_electric"):
            self.im_electric = value
        elif key == "d_q":
            self.d_q = value
        elif key in ("d_q*", "d_q_conj"):
            self.d_q_conj = value
        else:
            raise KeyError(f"Invalid key '{key}' for OBEEvolutionMatrices")

    def __delitem__(self, key: Union[str, TransitionKey]) -> None:
        if key == "d_q":
            self.d_q = {}
        elif key in ("d_q*", "d_q_conj"):
            self.d_q_conj = {}
        else:
            raise KeyError(f"Cannot delete key '{key}' from OBEEvolutionMatrices")

    def __contains__(self, key: str) -> bool:
        valid_keys = {"decay", "H0", "h0", "static", "B", "magnetic", "reE", "re_electric", "imE",
                      "im_electric", "d_q", "d_q*", "d_q_conj"}
        return key in valid_keys

    def keys(self) -> list[str]:
        keys_list = ["decay", "H0", "static", "B"]
        if self.re_electric:
            keys_list.append("reE")
        if self.im_electric:
            keys_list.append("imE")
        if self.d_q:
            keys_list.append("d_q")
        if self.d_q_conj:
            keys_list.append("d_q*")
        return keys_list

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())


@dataclass
class RateEqEvolutionMatrices:
    """
    Structured container for Rate Equation evolution matrices.

    Provides attribute access (.decay, .pumping) and dictionary indexing (['decay'], ['pumping']).
    """
    decay: np.ndarray = field(default_factory=lambda: np.array([]))
    pumping: dict[TransitionKey, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key == "decay":
            return self.decay
        elif key == "pumping":
            return self.pumping
        raise KeyError(f"Invalid key '{key}' for RateEqEvolutionMatrices")

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "decay":
            self.decay = value
        elif key == "pumping":
            self.pumping = value
        else:
            raise KeyError(f"Invalid key '{key}' for RateEqEvolutionMatrices")

    def __contains__(self, key: str) -> bool:
        return key in ("decay", "pumping")

    def keys(self) -> list[str]:
        return ["decay", "pumping"]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())


# endregion

def sort_float_then_string(numbers, strings):
    energy_label_pairs = np.asarray([(numbers[i], strings[i]) for i in range(len(numbers))],
                                    dtype=[('energy', float), ('label', 'S10')])
    return np.argsort(energy_label_pairs, order=['energy', 'label'])
