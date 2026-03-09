"""
models/quantum/vqc_amplitude.py

Variational Quantum Classifier using amplitude encoding.

Amplitude encoding maps n features into log2(n) qubits by encoding
the feature vector as amplitudes of the quantum state:
    |ψ(x)⟩ = Σ_i (x_i / ||x||) |i⟩

This is more qubit-efficient but requires state preparation circuits
that scale exponentially in depth. Here we use PennyLane's
qml.AmplitudeEmbedding which handles the state prep automatically.

Key difference from angle encoding:
  - Requires fewer qubits (ceil(log2(n_features)) instead of n_features)
  - Encodes relative magnitudes, not absolute values
  - Normalizes inputs: bias implications are different

For 7 features → 3 qubits (pad to 8 features)
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
import math


class VQCAmplitude(BaseEstimator, ClassifierMixin):
    """
    VQC with amplitude encoding.

    Args:
        n_features   : Number of input features (will be padded to next power of 2)
        n_layers     : Number of ansatz layers
        n_epochs     : Training epochs
        lr           : Adam learning rate
        batch_size   : Mini-batch size
        device       : PennyLane device
        shots        : Measurement shots (None = exact)
        random_state : Seed
    """

    def __init__(
        self,
        n_features: int = 7,
        n_layers: int = 3,
        n_epochs: int = 60,
        lr: float = 0.02,
        batch_size: int = 32,
        device: str = "default.qubit",
        shots: int = None,
        random_state: int = 42,
    ):
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.shots = shots
        self.random_state = random_state

        # Compute qubit count: ceil(log2(n_features)) but at least 2
        self.n_qubits = max(2, math.ceil(math.log2(n_features)))
        self.pad_to = 2 ** self.n_qubits
        self.weights_ = None
        self.loss_history_ = []

    def _pad(self, X: np.ndarray) -> np.ndarray:
        """Pad feature vectors to length 2^n_qubits and L2-normalize."""
        n, d = X.shape
        if d < self.pad_to:
            X = np.hstack([X, np.zeros((n, self.pad_to - d))])
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div by zero
        return X / norms

    def _build_circuit(self):
        dev = qml.device(self.device, wires=self.n_qubits, shots=self.shots)

        @qml.qnode(dev, interface="autograd", diff_method="best")
        def circuit(weights, x):
            # Amplitude encoding: maps normalized x onto 2^n_qubits amplitudes
            qml.AmplitudeEmbedding(x, wires=range(self.n_qubits), normalize=False)

            # Hardware-efficient ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            return qml.expval(qml.PauliZ(0))

        return circuit

    def _init_weights(self):
        np.random.seed(self.random_state)
        return pnp.array(
            np.random.uniform(-np.pi, np.pi, (self.n_layers, self.n_qubits, 3)),
            requires_grad=True,
        )

    def _loss(self, circuit, weights, X_batch, y_batch):
        predictions = pnp.array([circuit(weights, x) for x in X_batch])
        probs = pnp.clip((predictions + 1) / 2, 1e-7, 1 - 1e-7)
        y = pnp.array(y_batch, dtype=float)
        return -pnp.mean(y * pnp.log(probs) + (1 - y) * pnp.log(1 - probs))

    def fit(self, X, y, X_val=None, y_val=None):
        X_enc = self._pad(X)
        X_val_enc = self._pad(X_val) if X_val is not None else None

        circuit = self._build_circuit()
        weights = self._init_weights()
        opt = qml.AdamOptimizer(stepsize=self.lr)

        n = len(X_enc)
        best_val = float("inf")
        self.val_loss_history_ = []

        for epoch in tqdm(range(self.n_epochs), desc="[VQC-Amp] Training"):
            perm = np.random.permutation(n)
            X_shuf, y_shuf = X_enc[perm], y[perm]

            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                Xb = X_shuf[start : start + self.batch_size]
                yb = y_shuf[start : start + self.batch_size]

                def cost(w):
                    return self._loss(circuit, w, Xb, yb)

                weights, lv = opt.step_and_cost(cost, weights)
                epoch_loss += lv

            self.loss_history_.append(float(epoch_loss))

            if X_val_enc is not None:
                vl = float(self._loss(circuit, weights, X_val_enc, y_val))
                self.val_loss_history_.append(vl)
                if vl < best_val:
                    best_val = vl
                    self._best_weights = weights.copy()

        self.weights_ = getattr(self, "_best_weights", weights)
        self._circuit = circuit
        self._X_enc = None  # don't store training data
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_enc = self._pad(X)
        scores = np.array([float(self._circuit(self.weights_, x)) for x in X_enc])
        probs = np.clip((scores + 1) / 2, 0.0, 1.0)
        return np.column_stack([1 - probs, probs])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def parameter_count(self) -> int:
        return int(np.prod((self.n_layers, self.n_qubits, 3)))
