"""
models/quantum/vqc_angle.py

Variational Quantum Classifier using angle (Pauli rotation) encoding.

Architecture:
  1. Data encoding: RY(x_i * pi) on each qubit (angle encoding)
  2. Ansatz: Strongly Entangling Layers (hardware-efficient)
  3. Measurement: <Z> on qubit 0, threshold at 0 for binary classification

Angle encoding is the most common QML encoding and the most studied
in the fairness/bias literature. We include it as our primary model.

Reference: Schuld et al. (2020). "Circuit-centric quantum classifiers."
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm


class VQCAngle(BaseEstimator, ClassifierMixin):
    """
    Variational Quantum Classifier with angle encoding.

    Args:
        n_qubits     : Number of qubits (= number of input features after PCA/selection)
        n_layers     : Number of strongly entangling ansatz layers
        n_epochs     : Optimization epochs
        lr           : Adam learning rate
        batch_size   : Mini-batch size for gradient estimation
        device       : PennyLane device string ("default.qubit" or "qiskit.aer")
        shots        : None for exact simulation, int for shot-based
        random_state : Seed
    """

    def __init__(
        self,
        n_qubits: int = 7,
        n_layers: int = 3,
        n_epochs: int = 60,
        lr: float = 0.02,
        batch_size: int = 32,
        device: str = "default.qubit",
        shots: int = None,
        random_state: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.shots = shots
        self.random_state = random_state
        self.weights_ = None
        self.loss_history_ = []

    def _build_device(self):
        return qml.device(self.device, wires=self.n_qubits, shots=self.shots)

    def _build_circuit(self, dev):
        @qml.qnode(dev, interface="autograd", diff_method="best")
        def circuit(weights, x):
            # --- Angle Encoding ---
            # Each feature x_i is encoded as RY rotation on qubit i
            # This is a Pauli feature map: |x⟩ = ⊗_i RY(π·x_i)|0⟩
            for i in range(self.n_qubits):
                qml.RY(np.pi * x[i], wires=i)

            # --- Strongly Entangling Ansatz ---
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            # --- Measurement: expectation value of Z on qubit 0 ---
            return qml.expval(qml.PauliZ(0))

        return circuit

    def _init_weights(self):
        np.random.seed(self.random_state)
        # Shape: (n_layers, n_qubits, 3) — 3 Euler angles per qubit per layer
        return pnp.array(
            np.random.uniform(-np.pi, np.pi, (self.n_layers, self.n_qubits, 3)),
            requires_grad=True,
        )

    def _loss(self, circuit, weights, X_batch, y_batch):
        """Binary cross-entropy loss using quantum circuit predictions."""
        predictions = pnp.array([circuit(weights, x) for x in X_batch])
        # Map [-1, 1] expectation → [0, 1] probability
        probs = (predictions + 1) / 2
        probs = pnp.clip(probs, 1e-7, 1 - 1e-7)
        y = pnp.array(y_batch, dtype=float)
        bce = -pnp.mean(y * pnp.log(probs) + (1 - y) * pnp.log(1 - probs))
        return bce

    def fit(self, X, y, X_val=None, y_val=None):
        np.random.seed(self.random_state)
        dev = self._build_device()
        circuit = self._build_circuit(dev)
        weights = self._init_weights()
        opt = qml.AdamOptimizer(stepsize=self.lr)

        n = len(X)
        best_val_loss = float("inf")
        self.val_loss_history_ = []

        for epoch in tqdm(range(self.n_epochs), desc="[VQC-Angle] Training"):
            # Mini-batch shuffle
            perm = np.random.permutation(n)
            X_shuf, y_shuf = X[perm], y[perm]

            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                Xb = X_shuf[start : start + self.batch_size]
                yb = y_shuf[start : start + self.batch_size]

                def cost(w):
                    return self._loss(circuit, w, Xb, yb)

                weights, loss_val = opt.step_and_cost(cost, weights)
                epoch_loss += loss_val

            self.loss_history_.append(float(epoch_loss))

            if X_val is not None:
                val_loss = float(self._loss(circuit, weights, X_val, y_val))
                self.val_loss_history_.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._best_weights = weights.copy()

        self.weights_ = getattr(self, "_best_weights", weights)
        self._circuit = circuit
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = np.array([float(self._circuit(self.weights_, x)) for x in X])
        probs = (scores + 1) / 2  # [-1,1] → [0,1]
        probs = np.clip(probs, 0.0, 1.0)
        return np.column_stack([1 - probs, probs])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def get_circuit_diagram(self):
        """Print human-readable circuit diagram for a single sample."""
        if self.weights_ is None:
            raise RuntimeError("Fit the model first.")
        dev = self._build_device()
        circuit = self._build_circuit(dev)
        dummy_x = np.zeros(self.n_qubits)
        print(qml.draw(circuit)(self.weights_, dummy_x))

    def parameter_count(self) -> int:
        return int(np.prod((self.n_layers, self.n_qubits, 3)))
