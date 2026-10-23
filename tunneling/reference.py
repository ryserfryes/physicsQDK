import numpy as np

from .model import Parameters, initial_amplitudes, potential


def _solve_tridiagonal(rhs, multipliers, pivots, off_diagonal):
    solution = rhs.copy()
    for i in range(1, solution.size):
        solution[i] -= multipliers[i - 1] * solution[i - 1]
    solution[-1] /= pivots[-1]
    for i in range(solution.size - 2, -1, -1):
        solution[i] = (solution[i] - off_diagonal * solution[i + 1]) / pivots[i]
    return solution


def simulate(p: Parameters) -> np.ndarray:
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
            barrier = potential(p)
            if p.barrier_height > 0 and not np.any(barrier):
                raise ValueError("Barrier falls between grid points; increase n_qubits")

            kinetic_scale = p.dt * p.hbar / (p.mass * np.square(p.dx))
            diagonal = 1 + 0.5j * (kinetic_scale + p.dt * barrier / p.hbar)
            off_diagonal = -0.25j * kinetic_scale
            gamma = -diagonal[0]
            corner_ratio = off_diagonal / gamma
            pivots = diagonal.copy()
            pivots[0] -= gamma
            pivots[-1] -= off_diagonal * corner_ratio
            multipliers = np.empty(p.size - 1, dtype=np.complex128)
            for i in range(1, p.size):
                multipliers[i - 1] = off_diagonal / pivots[i - 1]
                pivots[i] -= multipliers[i - 1] * off_diagonal

            corner = np.zeros(p.size, dtype=np.complex128)
            corner[0], corner[-1] = gamma, off_diagonal
            correction = _solve_tridiagonal(corner, multipliers, pivots, off_diagonal)
            denominator = 1 + correction[0] + corner_ratio * correction[-1]
            if not np.isfinite(denominator) or denominator == 0:
                raise ValueError("Cannot solve the periodic evolution system")

            right_diagonal = 2 - diagonal
            states = np.empty((p.steps + 1, p.size), dtype=np.complex128)
            states[0] = initial_amplitudes(p)
            for i in range(1, p.steps + 1):
                previous = states[i - 1]
                rhs = right_diagonal * previous - off_diagonal * (
                    np.roll(previous, 1) + np.roll(previous, -1)
                )
                solution = _solve_tridiagonal(rhs, multipliers, pivots, off_diagonal)
                factor = (solution[0] + corner_ratio * solution[-1]) / denominator
                states[i] = solution - factor * correction
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise ValueError("Cannot compute finite evolution for these parameters") from error

    if not np.all(np.isfinite(states)):
        raise ValueError("Evolution produced nonfinite amplitudes")
    return states
