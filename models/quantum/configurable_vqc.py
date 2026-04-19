"""
models/quantum/configurable_vqc.py

Single Variational Quantum Classifier with pluggable encoding, ansatz, and measurement.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

from models.quantum.ansatze import get_ansatz
from models.quantum.encodings import EncodingSpec, resolve_encoding

_NOISE_TYPES = ("fixed", "random")


def _measurement_op(measurement: str, n_qubits: int):
    measurement = measurement.lower()
    if measurement in ("z0", "z_0", "pauliz_0"):
        return qml.expval(qml.PauliZ(0))
    if measurement in ("zz_01", "zz01", "pauliz_zz"):
        if n_qubits < 2:
            raise ValueError("Measurement zz_01 requires at least 2 qubits.")
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    raise ValueError(
        f"Unknown measurement {measurement!r}. Use 'z0' or 'zz_01'."
    )


class ConfigurableVQC(BaseEstimator, ClassifierMixin):
    """
    VQC with configurable encoding, ansatz, and readout.

    Args:
        encoding: Name registered in encodings (e.g. angle, amplitude, iqp).
        ansatz: Name registered in ansatze (e.g. strongly_entangling, basic_entangler).
        measurement: 'z0' or 'zz_01' (IQP-style correlated readout).
        n_layers: Variational layers.
        n_epochs: Training epochs.
        lr: Adam step size.
        batch_size: Mini-batch size.
        device: PennyLane device string.
        shots: None for exact simulation, int for finite shots.
        random_state: RNG seed.
        user_n_qubits: For angle/iqp: override qubit count (first n_qubits features used).
        n_encoding_reps: IQP encoding repetitions.
        n_features_expected: Input column count for parameter_count() before fit (required for amplitude).
        model_label: Prefix for tqdm progress bar.
    """

    def __init__(
        self,
        encoding: str = "angle",
        ansatz: str = "strongly_entangling",
        measurement: str = "z0",
        n_layers: int = 3,
        n_epochs: int = 60,
        lr: float = 0.02,
        batch_size: int = 32,
        device: str = "default.qubit",
        shots: Optional[int] = None,
        random_state: int = 42,
        user_n_qubits: Optional[int] = None,
        n_encoding_reps: int = 2,
        n_features_expected: Optional[int] = None,
        model_label: str = "VQC",
        noise_type: Optional[str] = None,
        noise_strength: float = 0.0,
    ):
        if noise_type is not None and noise_type not in _NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {_NOISE_TYPES} or None, got {noise_type!r}")
        self.encoding = encoding
        self.ansatz = ansatz
        self.measurement = measurement
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.shots = shots
        self.random_state = random_state
        self.user_n_qubits = user_n_qubits
        self.n_encoding_reps = n_encoding_reps
        self.n_features_expected = n_features_expected
        self.model_label = model_label
        self.noise_type = noise_type
        self.noise_strength = noise_strength

        self.weights_ = None
        self.loss_history_: list = []
        self.encoding_spec_: Optional[EncodingSpec] = None
        self.n_features_: Optional[int] = None

    def _build_device(self, n_qubits: int):
        return qml.device(self.device, wires=n_qubits, shots=self.shots)

    def _build_circuit(self, dev, enc: EncodingSpec):
        ans = get_ansatz(self.ansatz)
        measurement = self.measurement
        noise_type = self.noise_type
        n_qubits = enc.n_qubits

        # Mutable holder so the training loop can update noise strength per batch.
        noise_p = [0.0]

        @qml.qnode(dev, interface="autograd", diff_method="best")
        def circuit(weights, x):
            enc.apply_encoding_circuit(x)
            ans.apply(weights, n_qubits)
            if noise_type:
                for wire in range(n_qubits):
                    qml.RX(noise_p[0] * np.pi, wires=wire)
                    qml.RZ(noise_p[0] * np.pi, wires=wire)
            # Measurement must be constructed inside the qfunc (not captured from outside).
            return _measurement_op(measurement, n_qubits)

        # Expose the noise holder so _loss can update it.
        circuit._noise_p = noise_p
        return circuit

    def _init_weights(self, n_qubits: int):
        rng = np.random.RandomState(self.random_state)
        ans = get_ansatz(self.ansatz)
        return ans.init_weights(self.n_layers, n_qubits, rng)

    def _set_noise_p(self, circuit, rng: np.random.RandomState):
        """Update the circuit's depolarizing noise probability for the current batch."""
        if self.noise_type == "fixed":
            circuit._noise_p[0] = float(self.noise_strength)
        elif self.noise_type == "random":
            circuit._noise_p[0] = float(rng.uniform(0.0, self.noise_strength))

    def _loss(self, circuit, weights, X_batch, y_batch):
        predictions = circuit(weights, X_batch)
        probs = (predictions + 1) / 2
        probs = pnp.clip(probs, 1e-7, 1 - 1e-7)
        y = pnp.array(y_batch, dtype=float)
        return -pnp.mean(y * pnp.log(probs) + (1 - y) * pnp.log(1 - probs))

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float64)
        self.n_features_ = X.shape[1]
        np.random.seed(self.random_state)

        enc = resolve_encoding(
            self.encoding,
            self.n_features_,
            n_qubits=self.user_n_qubits,
            n_encoding_reps=self.n_encoding_reps,
        )
        self.encoding_spec_ = enc

        X_train = enc.preprocess(X)
        X_val_p = enc.preprocess(X_val) if X_val is not None else None

        dev = self._build_device(enc.n_qubits)
        circuit = self._build_circuit(dev, enc)
        weights = self._init_weights(enc.n_qubits)
        opt = qml.AdamOptimizer(stepsize=self.lr)

        rng = np.random.RandomState(self.random_state)
        n = len(X_train)
        best_val = float("inf")
        self.val_loss_history_ = []

        if self.noise_type:
            print(f"  Noise: {self.noise_type}, strength={self.noise_strength}")

        desc = f"[{self.model_label}] Training"
        for epoch in tqdm(range(self.n_epochs), desc=desc):
            perm = np.random.permutation(n)
            X_shuf, y_shuf = X_train[perm], y[perm]

            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                Xb = X_shuf[start : start + self.batch_size]
                yb = y_shuf[start : start + self.batch_size]

                self._set_noise_p(circuit, rng)

                def cost(w):
                    return self._loss(circuit, w, Xb, yb)

                weights, lv = opt.step_and_cost(cost, weights)
                epoch_loss += lv

            self.loss_history_.append(float(epoch_loss))

            if X_val_p is not None:
                self._set_noise_p(circuit, rng)
                vl = float(self._loss(circuit, weights, X_val_p, y_val))
                self.val_loss_history_.append(vl)
                if vl < best_val:
                    best_val = vl
                    self._best_weights = weights.copy()

        self.weights_ = getattr(self, "_best_weights", weights)
        self._circuit = circuit
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.weights_ is None or self.encoding_spec_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self.noise_type:
            rng = np.random.RandomState(self.random_state)
            self._set_noise_p(self._circuit, rng)
        X_enc = self.encoding_spec_.preprocess(np.asarray(X, dtype=np.float64))
        scores = np.array(self._circuit(self.weights_, X_enc), dtype=float)
        probs = np.clip((scores + 1) / 2, 0.0, 1.0)
        return np.column_stack([1 - probs, probs])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def parameter_count(self) -> int:
        ans = get_ansatz(self.ansatz)
        if self.encoding_spec_ is not None:
            return ans.parameter_count(self.n_layers, self.encoding_spec_.n_qubits)
        nf = self.n_features_ or self.n_features_expected
        if nf is None:
            raise RuntimeError(
                "Call fit() first, or pass n_features_expected=... (input dimension)."
            )
        enc = resolve_encoding(
            self.encoding,
            nf,
            n_qubits=self.user_n_qubits,
            n_encoding_reps=self.n_encoding_reps,
        )
        return ans.parameter_count(self.n_layers, enc.n_qubits)

    def get_circuit_diagram(self):
        if self.weights_ is None or self.encoding_spec_ is None:
            raise RuntimeError("Fit the model first.")
        enc = self.encoding_spec_
        dev = self._build_device(enc.n_qubits)
        circuit = self._build_circuit(dev, enc)
        if enc.name == "amplitude":
            dummy = enc.preprocess(np.zeros((1, self.n_features_), dtype=np.float64))[0]
        else:
            dummy = np.zeros(enc.n_qubits, dtype=np.float64)
        print(qml.draw(circuit)(self.weights_, dummy))
