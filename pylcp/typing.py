from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union, Iterator
import numpy as np

class Signature(Enum):
    """Function argument signatures used for field promotion and lambdas."""
    POSITION_AND_TIME = "Rt"
    TIME_ONLY = "t"
    POSITION_ONLY = "R"

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
            raise ValueError(f"Invalid transition key string format: '{key_str}'. Expected 'ground->excited'.")
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
        valid_keys = {"decay", "H0", "h0", "B", "magnetic", "reE", "re_electric", "imE", "im_electric", "d_q", "d_q*", "d_q_conj"}
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
