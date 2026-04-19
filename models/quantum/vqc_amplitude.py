"""
models/quantum/vqc_amplitude.py

Variational Quantum Classifier using amplitude encoding.

Thin preset: amplitude encoding + strongly entangling + Z readout.
"""

from __future__ import annotations

import math

from models.quantum.configurable_vqc import ConfigurableVQC


class VQCAmplitude(ConfigurableVQC):
    """
    VQC with amplitude encoding.

    Args:
        n_features: Number of input columns (determines qubit count via ceil(log2))
        n_layers, n_epochs, lr, batch_size, device, shots, random_state: training hyperparams
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
        noise_type: str = None,
        noise_strength: float = 0.0,
    ):
        self._n_features_init = n_features
        super().__init__(
            encoding="amplitude",
            ansatz="strongly_entangling",
            measurement="z0",
            n_layers=n_layers,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            shots=shots,
            random_state=random_state,
            n_features_expected=n_features,
            model_label="VQC-Amp",
            noise_type=noise_type,
            noise_strength=noise_strength,
        )

    @property
    def n_features(self) -> int:
        return self.n_features_ if self.n_features_ is not None else self._n_features_init

    @property
    def n_qubits(self) -> int:
        if self.encoding_spec_ is not None:
            return self.encoding_spec_.n_qubits
        nq = max(2, int(math.ceil(math.log2(max(1, self._n_features_init)))))
        return nq

    @property
    def pad_to(self) -> int:
        return 2 ** self.n_qubits
