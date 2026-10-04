from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Union, Iterator, Literal, Callable, TYPE_CHECKING
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
                func(0)
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
    else: # 2 params
        try:
            func(np.zeros(3), 0)
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


class ValidationType(Enum):
    VectorLike = auto()
    NumericScalar = auto()


def validate_field_param_value(val: NumericScalar | VectorLike,
                               name: str, validation_type: ValidationType, error_text_for_callable):

    if validation_type == ValidationType.NumericScalar:
        # check if type of constant is correct
        if not isinstance(val, float | complex):
            raise TypeError(f"{name.capitalize()} " +
                          f"{"function must return" if error_text_for_callable else "must be"}" +
                          f" a float or complex number")
        return val
    else:  # validation_type == ValidationType.VectorLike
        # check if type of vector is correct
        if not isinstance(val, VectorLike):
            raise TypeError(f"{name.capitalize()} " +
                            f"{"function must return" if error_text_for_callable else "be"}" +
                            " an array-like of 3 float or complex numbers")
        # region check if size of vector is correct
        not3error = ValueError(f"{name.capitalize()} " +
                               f"{"function must return" if error_text_for_callable else "be"}" +
                               f" a vector with 3 components")
        if isinstance(val, (list, tuple)):
            if len(val) != 3:
                raise not3error
            return np.asarray(val)
        else:
            if TYPE_CHECKING:
                assert isinstance(val, np.ndarray)
            if val.shape[0] != 3 or len(val.shape) != 1:  # TODO: this check may be wrong
                raise not3error
            return val
        # endregion
        # TODO: check datatype?


class FieldParameter:
    """
    Wraps a scalar, vector, or function parameter with explicit dependency metadata.
    Avoids unnecessary lambda wrapping when the parameter is constant.
    """
    val: FieldParameterValue
    name: str
    dependence: Dependence

    def __init__(
            self, val: FieldParameterValue, name: str,validation_type: ValidationType):
        self.name = name

        if not callable(val):
            self.dependence = Dependence.CONSTANT
            self.val = validate_field_param_value(val, name, validation_type, False)
        else:
            self.val = val
            self.dependence = _inspect_callable_dependence(val)
            # region validate func returns at zeroes
            if self.dependence == Dependence.POSITION_ONLY:
                validate_field_param_value(self.val(np.zeros(3)), name, validation_type, True)
            elif self.dependence == Dependence.TIME_ONLY:
                validate_field_param_value(0., name, validation_type, True)
            else:
                validate_field_param_value(self.val(np.zeros(3),0.), name, validation_type, True)
            # endregion

    def __call__(self, R=np.array([0., 0., 0.]), t=0.):
        if self.dependence == Dependence.CONSTANT:
            return self.val

        if TYPE_CHECKING:
            assert callable(self.val)

        if self.dependence == Dependence.POSITION_ONLY:
            return self.val(R)
        elif self.dependence == Dependence.TIME_ONLY:
            return self.val(t)
        else:
            return self.val(R, t)


@dataclass(frozen=True, slots=True)
class TransitionKey:
    """Strongly typed representation of an atomic transition key (e.g. g->e)."""
    ground: str
    excited: str

    def __str__(self) -> str:
        return f"{self.ground}->{self.excited}"

    @classmethod
    def from_string(cls, key_str: str) -> "TransitionKey":
        if "->" not in key_str:
            raise ValueError(
                f"Invalid transition key string format: '{key_str}'. Expected 'ground->excited'.")
        g, e = key_str.split("->", 1)
        return cls(ground=g, excited=e)


@dataclass
class OBEEvolutionMatrices:
    """
    Structured container for Optical Bloch Equation evolution matrices.
    
    Provides both attribute access (.decay, .h0, .magnetic, .re_electric, .im_electric, .d_q, .d_q_conj)
    and dictionary-compatible indexing (['decay'], ['H0'], ['B'], ['reE'], ['imE'], ['d_q'], ['d_q*']).
    """
    decay: np.ndarray = field(default_factory=lambda: np.array([]))
    h0: np.ndarray = field(default_factory=lambda: np.array([]))
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
        valid_keys = {"decay", "H0", "h0", "B", "magnetic", "reE", "re_electric", "imE",
                      "im_electric", "d_q", "d_q*", "d_q_conj"}
        return key in valid_keys

    def keys(self) -> list[str]:
        keys_list = ["decay", "H0", "B"]
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
