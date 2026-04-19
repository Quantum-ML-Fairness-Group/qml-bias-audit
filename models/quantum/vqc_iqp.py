"""
models/quantum/vqc_iqp.py

Variational Quantum Classifier using IQP encoding.

Thin preset: IQP feature map + strongly entangling + ZZ readout on qubits 0,1.

Reference:
  Havlíček et al. (2019). "Supervised learning with quantum-enhanced feature spaces."
"""

from __future__ import annotations

from models.quantum.configurable_vqc import ConfigurableVQC


class VQCIQP(ConfigurableVQC):
    """
    VQC with IQP feature map.

    Args:
        n_qubits: Number of qubits (= first n_qubits features used)
        n_encoding_reps: Times to repeat the IQP encoding block
        lr: Default 0.015 (slightly lower than angle/amplitude)
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
        noise_type: str = None,
        noise_strength: float = 0.0,
    ):
        super().__init__(
            encoding="iqp",
            ansatz="strongly_entangling",
            measurement="zz_01",
            n_layers=n_layers,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            shots=shots,
            random_state=random_state,
            user_n_qubits=n_qubits,
            n_encoding_reps=n_encoding_reps,
            n_features_expected=n_qubits,
            model_label="VQC-IQP",
            noise_type=noise_type,
            noise_strength=noise_strength,
        )

    def circuit_info(self):
        """Return circuit metadata."""
        nq = self.encoding_spec_.n_qubits if self.encoding_spec_ else self.user_n_qubits
        return {
            "n_qubits": nq,
            "n_layers": self.n_layers,
            "n_encoding_reps": self.n_encoding_reps,
            "n_parameters": self.parameter_count(),
            "encoding": "IQP",
        }
