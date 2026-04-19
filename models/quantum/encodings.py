"""
models/quantum/encodings.py

Registered data encodings for VQCs. Each encoding defines:
  - how many qubits are needed for a given number of input features
  - optional batch preprocessing (padding, slicing, normalization)
  - PennyLane queue operations inside the QNode

See EncodingSpec dataclass for the full contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pennylane as qml


def iqp_encoding_block(x: np.ndarray, n_qubits: int) -> None:
    """One repetition of the IQP feature map (Havlíček et al., 2019)."""
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    for i in range(n_qubits):
        qml.RZ(x[..., i], wires=i)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            qml.CNOT(wires=[i, j])
            qml.RZ(x[..., i] * x[..., j], wires=j)
            qml.CNOT(wires=[i, j])


def apply_angle_encoding(x: np.ndarray, n_qubits: int) -> None:
    """RY(π·x_i) on each wire i (x length >= n_qubits)."""
    for i in range(n_qubits):
        qml.RY(np.pi * x[..., i], wires=i)


def apply_amplitude_encoding(x: np.ndarray, n_qubits: int) -> None:
    """Amplitude embedding; x must have length 2**n_qubits, already normalized."""
    qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=False)


def apply_iqp_encoding(x: np.ndarray, n_qubits: int, n_reps: int) -> None:
    for _ in range(n_reps):
        iqp_encoding_block(x, n_qubits)


def preprocess_angle(X: np.ndarray, n_qubits: int) -> np.ndarray:
    """Use first n_qubit features per row (pad with zeros if fewer columns)."""
    n, d = X.shape
    if d < n_qubits:
        X = np.hstack([X, np.zeros((n, n_qubits - d), dtype=X.dtype)])
    return X[:, :n_qubits].astype(np.float64)


def preprocess_iqp(X: np.ndarray, n_qubits: int) -> np.ndarray:
    """Same as angle: first n_qubits features."""
    return preprocess_angle(X, n_qubits)


ENCODING_REGISTRY: Dict[str, str] = {
    "angle": "Angle / Pauli-RY encoding",
    "amplitude": "Amplitude embedding (padded to power of 2, L2-normalized)",
    "iqp": "IQP feature map (Havlíček et al.)",
}


def preprocess_amplitude(X: np.ndarray, n_features: int) -> tuple[np.ndarray, int, int]:
    """
    Pad to next power of 2, L2-normalize per row.

    Returns:
        X_out: shape (n, pad_to)
        n_qubits: wire count
        pad_to: 2**n_qubits
    """
    n_qubits = max(2, int(math.ceil(math.log2(max(1, n_features)))))
    pad_to = 2**n_qubits
    n, d = X.shape
    if d < pad_to:
        X = np.hstack([X, np.zeros((n, pad_to - d), dtype=np.float32)])
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (X / norms).astype(np.float64), n_qubits, pad_to


@dataclass
class EncodingSpec:
    """
    Resolved encoding for a fixed input dimension n_features.

    Attributes:
        name: Registry key ("angle", "amplitude", "iqp").
        n_qubits: Number of quantum wires.
        n_encoding_reps: IQP repetitions only (else ignored).
        preprocess: Maps design matrix X -> array passed row-wise to the QNode.
    """

    name: str
    n_qubits: int
    n_encoding_reps: int
    preprocess: Callable[[np.ndarray], np.ndarray]

    def apply_encoding_circuit(self, x: np.ndarray) -> None:
        """Apply the correct PennyLane encoding (single sample x)."""
        if self.name == "angle":
            apply_angle_encoding(x, self.n_qubits)
        elif self.name == "amplitude":
            apply_amplitude_encoding(x, self.n_qubits)
        elif self.name == "iqp":
            apply_iqp_encoding(x, self.n_qubits, self.n_encoding_reps)
        else:
            raise ValueError(f"Unknown encoding: {self.name}")


def resolve_encoding(
    name: str,
    n_features: int,
    *,
    n_qubits: Optional[int] = None,
    n_encoding_reps: int = 2,
) -> EncodingSpec:
    """
    Build an EncodingSpec for n_features input columns.

    Args:
        name: "angle" | "amplitude" | "iqp"
        n_features: Number of columns in X (before preprocess).
        n_qubits: For angle/iqp, defaults to n_features; may be smaller (first n_qubits used).
        n_encoding_reps: For iqp only.
    """
    name = name.lower()
    if name == "angle":
        nq = n_qubits if n_qubits is not None else n_features

        def pre(X: np.ndarray) -> np.ndarray:
            return preprocess_angle(X, nq)

        return EncodingSpec(
            name=name,
            n_qubits=nq,
            n_encoding_reps=1,
            preprocess=pre,
        )

    if name == "amplitude":

        def pre_amp(X: np.ndarray) -> np.ndarray:
            out, _, _ = preprocess_amplitude(X, n_features)
            return out

        _, nq, _ = preprocess_amplitude(np.zeros((1, n_features), dtype=np.float32), n_features)

        return EncodingSpec(
            name=name,
            n_qubits=nq,
            n_encoding_reps=1,
            preprocess=pre_amp,
        )

    if name == "iqp":
        nq = n_qubits if n_qubits is not None else n_features

        def pre_iqp(X: np.ndarray) -> np.ndarray:
            return preprocess_iqp(X, nq)

        return EncodingSpec(
            name=name,
            n_qubits=nq,
            n_encoding_reps=n_encoding_reps,
            preprocess=pre_iqp,
        )

    raise ValueError(f"Unknown encoding {name!r}. Choose from: {list(ENCODING_REGISTRY.keys())}")


def list_encodings() -> List[str]:
    return list(ENCODING_REGISTRY.keys())
