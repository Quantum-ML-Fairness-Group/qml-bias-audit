"""
models/quantum/vqc_iqp.py

Variational Quantum Classifier using IQP (Instantaneous Quantum Polynomial) encoding.

IQP encoding is theoretically motivated: it creates feature maps that are
classically hard to simulate under plausible complexity-theoretic assumptions
(Havlíček et al., 2019). The encoding consists of:

  1. Hadamard layer (puts all qubits in superposition)
  2. Data-dependent Z rotations: RZ(x_i) on each qubit
  3. Data-dependent ZZ interactions: CNOT + RZ(x_i * x_j) + CNOT (second-order terms)
  4. Repeat encoding block `n_reps` times for richer expressibility
  5. Parameterized ansatz (strongly entangling)
  6. Measurement: <Z⊗Z> on qubits 0,1 (reduces variance vs single-qubit)

This is closest to the Havlíček et al. quantum kernel SVM, but in the VQC
setting where we optimize the ansatz parameters end-to-end.

Reference:
  Havlíček et al. (2019). "Supervised learning with quantum-enhanced feature spaces."
  Nature 567, 209–212.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm


def _iqp_encoding_block(x, n_qubits: int):
    """
    One repetition of IQP feature map.
    x shape: (n_qubits,) — one feature per qubit
    """
    # Layer 1: Hadamards
    for i in range(n_qubits):
        qml.Hadamard(wires=i)

    # Layer 2: First-order RZ
    for i in range(n_qubits):
        qml.RZ(x[i], wires=i)

    # Layer 3: Second-order ZZ interactions (all pairs)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            qml.CNOT(wires=[i, j])
            qml.RZ(x[i] * x[j], wires=j)  # ZZ term
            qml.CNOT(wires=[i, j])


class VQCIQP(BaseEstimator, ClassifierMixin):
    """
    VQC with IQP feature map.

    Args:
        n_qubits     : Number of qubits (= n_features used; set <= actual features)
        n_layers     : Ansatz depth
        n_encoding_reps : Times to repeat IQP encoding block (1 or 2 recommended)
        n_epochs     : Training epochs
        lr           : Adam learning rate
        batch_size   : Mini-batch
        device       : PennyLane device
        shots        : Measurement shots
        random_state : Seed
    """

    def __init__(
        self,
        n_qubits: int = 7,
        n_layers: int = 3,
        n_encoding_reps: int = 2,
        n_epochs: int = 60,
        lr: float = 0.015,
        batch_size: int = 32,
        device: str = "default.qubit",
        shots: int = None,
        random_state: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_encoding_reps = n_encoding_reps
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.shots = shots
        self.random_state = random_state
        self.weights_ = None
        self.loss_history_ = []

    def _build_circuit(self):
        dev = qml.device(self.device, wires=self.n_qubits, shots=self.shots)

        @qml.qnode(dev, interface="autograd", diff_method="best")
        def circuit(weights, x):
            # --- IQP Feature Map (possibly repeated) ---
            for _ in range(self.n_encoding_reps):
                _iqp_encoding_block(x, self.n_qubits)

            # --- Variational Ansatz ---
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            # --- Measurement: tensor product ZZ on qubits 0,1 ---
            # Using ZZ expectation reduces measurement variance vs single Z
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

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
        # Use only first n_qubits features (select most informative)
        X_use = X[:, : self.n_qubits]
        X_val_use = X_val[:, : self.n_qubits] if X_val is not None else None

        circuit = self._build_circuit()
        weights = self._init_weights()
        opt = qml.AdamOptimizer(stepsize=self.lr)

        n = len(X_use)
        best_val = float("inf")
        self.val_loss_history_ = []

        for epoch in tqdm(range(self.n_epochs), desc="[VQC-IQP] Training"):
            perm = np.random.permutation(n)
            X_shuf, y_shuf = X_use[perm], y[perm]

            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                Xb = X_shuf[start : start + self.batch_size]
                yb = y_shuf[start : start + self.batch_size]

                def cost(w):
                    return self._loss(circuit, w, Xb, yb)

                weights, lv = opt.step_and_cost(cost, weights)
                epoch_loss += lv

            self.loss_history_.append(float(epoch_loss))

            if X_val_use is not None:
                vl = float(self._loss(circuit, weights, X_val_use, y_val))
                self.val_loss_history_.append(vl)
                if vl < best_val:
                    best_val = vl
                    self._best_weights = weights.copy()

        self.weights_ = getattr(self, "_best_weights", weights)
        self._circuit = circuit
        self._n_qubits_used = self.n_qubits
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_use = X[:, : self._n_qubits_used]
        scores = np.array([float(self._circuit(self.weights_, x)) for x in X_use])
        probs = np.clip((scores + 1) / 2, 0.0, 1.0)
        return np.column_stack([1 - probs, probs])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def circuit_info(self):
        """Return circuit depth and parameter count."""
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_encoding_reps": self.n_encoding_reps,
            "n_parameters": int(np.prod((self.n_layers, self.n_qubits, 3))),
            "encoding": "IQP",
        }
    
    def parameter_count(self) -> int:
        return int(np.prod((self.n_layers, self.n_qubits, 3)))
