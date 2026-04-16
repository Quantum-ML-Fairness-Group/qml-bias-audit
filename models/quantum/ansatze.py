"""
models/quantum/ansatze.py

Variational ansatz templates paired with weight initialization and parameter counts.
Each ansatz must be compatible with the ConfigurableVQC training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def apply_strongly_entangling(weights, n_qubits: int) -> None:
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))


def init_strongly_entangling(
    n_layers: int, n_qubits: int, rng: np.random.RandomState
):
    w = rng.uniform(-np.pi, np.pi, (n_layers, n_qubits, 3))
    return pnp.array(w, requires_grad=True)


def param_count_strongly_entangling(n_layers: int, n_qubits: int) -> int:
    return int(n_layers * n_qubits * 3)


def apply_basic_entangler(weights, n_qubits: int) -> None:
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))


def init_basic_entangler(
    n_layers: int, n_qubits: int, rng: np.random.RandomState
):
    w = rng.uniform(-np.pi, np.pi, (n_layers, n_qubits))
    return pnp.array(w, requires_grad=True)


def param_count_basic_entangler(n_layers: int, n_qubits: int) -> int:
    return int(n_layers * n_qubits)


@dataclass(frozen=True)
class AnsatzSpec:
    """Ansatz implementation bundle."""

    name: str
    apply: Callable[..., None]
    init_weights: Callable[..., pnp.ndarray]
    parameter_count: Callable[[int, int], int]


ANSATZE: Dict[str, AnsatzSpec] = {
    "strongly_entangling": AnsatzSpec(
        name="strongly_entangling",
        apply=apply_strongly_entangling,
        init_weights=init_strongly_entangling,
        parameter_count=param_count_strongly_entangling,
    ),
    "basic_entangler": AnsatzSpec(
        name="basic_entangler",
        apply=apply_basic_entangler,
        init_weights=init_basic_entangler,
        parameter_count=param_count_basic_entangler,
    ),
}


def get_ansatz(name: str) -> AnsatzSpec:
    key = name.lower()
    if key not in ANSATZE:
        raise ValueError(f"Unknown ansatz {name!r}. Choose from: {list(ANSATZE.keys())}")
    return ANSATZE[key]


def list_ansatze() -> list[str]:
    return list(ANSATZE.keys())
