"""
models/quantum/vqc_angle.py

Variational Quantum Classifier using angle (Pauli rotation) encoding.

This class is a thin preset over ConfigurableVQC (angle + strongly entangling + Z readout).

Reference: Schuld et al. (2020). "Circuit-centric quantum classifiers."
"""

from __future__ import annotations

from models.quantum.configurable_vqc import ConfigurableVQC


class VQCAngle(ConfigurableVQC):
    """
    Variational Quantum Classifier with angle encoding.

    Args:
        n_qubits: Number of qubits (= number of input features used, first n_qubits columns)
        n_layers: Number of strongly entangling ansatz layers
        n_epochs, lr, batch_size, device, shots, random_state: as in ConfigurableVQC
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
        noise_type: str = None,
        noise_strength: float = 0.0,
    ):
        super().__init__(
            encoding="angle",
            ansatz="strongly_entangling",
            measurement="z0",
            n_layers=n_layers,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            shots=shots,
            random_state=random_state,
            user_n_qubits=n_qubits,
            n_features_expected=n_qubits,
            model_label="VQC-Angle",
            noise_type=noise_type,
            noise_strength=noise_strength,
        )
