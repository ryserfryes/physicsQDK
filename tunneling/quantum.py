"""QDK bridge; imports are lazy so packet preview works without QDK."""
from pathlib import Path
import numpy as np


def sample_register(project_root: Path, n_qubits: int, shots: int) -> np.ndarray:
    from qdk import qsharp

    if type(n_qubits) is not int or not 1 <= n_qubits <= 12:
        raise ValueError("n_qubits must be an integer in 1..12")
    if type(shots) is not int or shots < 1:
        raise ValueError("shots must be a positive integer")
    qsharp.init(project_root=str(project_root))
    values = qsharp.run(f"Tunneling.SampleUniformPosition({n_qubits})", shots=shots)
    return np.asarray(values, dtype=np.int64)


def simulate(*args, **kwargs):
    """Future API: parameters -> snapshots and final R/T probabilities."""
    raise NotImplementedError("Implement state preparation and evolution in src/ first")
