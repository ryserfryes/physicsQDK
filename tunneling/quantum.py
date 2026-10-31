from pathlib import Path
import numpy as np

from .model import Parameters, initial_amplitudes


def sample_register(project_root: Path, n_qubits: int, shots: int) -> np.ndarray:
    from qdk import qsharp

    if type(n_qubits) is not int or not 1 <= n_qubits <= 12:
        raise ValueError("n_qubits must be an integer in 1..12")
    if type(shots) is not int or shots < 1:
        raise ValueError("shots must be a positive integer")
    qsharp.init(project_root=str(project_root))
    values = qsharp.run(f"Tunneling.SampleUniformPosition({n_qubits})", shots=shots)
    return np.asarray(values, dtype=np.int64)


def _qsharp_array(values):
    """Render finite Python floats as Q# array literals."""
    def literal(value):
        rendered = format(float(value), ".17g")
        return rendered if any(char in rendered for char in ".eE") else rendered + ".0"

    return "[" + ", ".join(literal(value) for value in values) + "]"


def _state_vector(qsharp, size):
    dump = qsharp.dump_machine()
    if dump.qubit_count != size.bit_length() - 1:
        raise ValueError("QDK state has an unexpected register size")
    amplitudes = np.zeros(size, dtype=np.complex128)
    for index in range(size):
        try:
            value = dump[index]
        except KeyError:
            continue  # StateDump stores only nonzero amplitudes.
        if value is not None:
            amplitudes[index] = complex(value)
    return amplitudes


def simulate_free(p: Parameters, snapshot_steps, project_root: Path) -> np.ndarray:
    from qdk import qsharp

    steps = tuple(snapshot_steps)
    if not steps or any(type(step) is not int or step < 0 for step in steps):
        raise ValueError("snapshot_steps must contain nonnegative integers")
    if tuple(sorted(set(steps))) != steps:
        raise ValueError("snapshot_steps must be sorted and unique")

    qsharp.init(project_root=str(project_root))
    envelope = np.abs(initial_amplitudes(p))
    position_weights = 2 ** np.arange(p.n_qubits - 1, -1, -1)
    position_phases = p.p0 * p.dx / p.hbar * position_weights
    momentum_weights = 2 ** np.arange(p.n_qubits, dtype=np.int64)
    momentum_weights[-1] *= -1
    phase_scale = -p.dt * p.hbar * (2 * np.pi / (p.x_max - p.x_min)) ** 2 / (2 * p.mass)
    bit_phases = phase_scale * momentum_weights ** 2
    pair_phases = 2 * phase_scale * np.outer(momentum_weights, momentum_weights)

    qsharp.eval(f"use position = Qubit[{p.n_qubits}];")
    qsharp.eval(
        "Tunneling.PrepareWavePacket(position, "
        f"{_qsharp_array(envelope)}, {_qsharp_array(position_phases)});"
    )
    step_expr = (
        "Tunneling.EvolutionStep(position, "
        f"{_qsharp_array(bit_phases)}, "
        "[" + ", ".join(_qsharp_array(row) for row in pair_phases) + "]);"
    )
    states = np.empty((len(steps), p.size), dtype=np.complex128)
    current = 0
    for index, target in enumerate(steps):
        for _ in range(target - current):
            qsharp.eval(step_expr)
        states[index] = _state_vector(qsharp, p.size)
        current = target
    return states
