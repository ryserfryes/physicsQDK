"""Shared spatial grid and initial conditions for both future solvers."""
from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np


@dataclass(frozen=True)
class Parameters:
    n_qubits: int = 8
    x_min: float = -20.0
    x_max: float = 20.0
    x0: float = -8.0
    sigma_x: float = 1.5
    p0: float = 1.0
    mass: float = 1.0
    hbar: float = 1.0
    barrier_center: float = 0.0
    barrier_width: float = 1.0
    barrier_height: float = 1.0
    dt: float = 0.02
    steps: int = 600
    shots: int = 2000

    def __post_init__(self):
        for key in ("n_qubits", "steps", "shots"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if not 1 <= self.n_qubits <= 12:
            raise ValueError("This starter supports 1..12 coordinate qubits")
        for key, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        if self.x_min >= self.x_max or not self.x_min < self.x0 < self.x_max:
            raise ValueError("Require x_min < x0 < x_max")
        if min(self.sigma_x, self.mass, self.hbar, self.dt, self.barrier_width) <= 0:
            raise ValueError("Widths, mass, hbar and dt must be positive")
        if self.barrier_height < 0 or self.steps < 1 or self.shots < 1:
            raise ValueError("Require nonnegative barrier height and positive steps/shots")
        if not self.x_min < self.barrier_left < self.barrier_right < self.x_max:
            raise ValueError("The barrier must fit inside the spatial domain")

    @property
    def size(self):
        return 2 ** self.n_qubits

    @property
    def dx(self):
        return (self.x_max - self.x_min) / self.size

    @property
    def barrier_left(self):
        return self.barrier_center - self.barrier_width / 2

    @property
    def barrier_right(self):
        return self.barrier_center + self.barrier_width / 2


def load_parameters(path: Path) -> Parameters:
    with path.open("rb") as stream:
        return Parameters(**tomllib.load(stream))


def make_grid(p: Parameters) -> np.ndarray:
    return p.x_min + p.dx * np.arange(p.size)


def initial_amplitudes(p: Parameters) -> np.ndarray:
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
            x = make_grid(p)
            log_envelope = -(x - p.x0) ** 2 / (4 * p.sigma_x ** 2)
            alpha = np.exp(log_envelope - np.max(log_envelope)) * np.exp(1j * p.p0 * x / p.hbar)
            norm = np.linalg.norm(alpha)
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("Initial packet must have a finite positive norm")
            return alpha / norm
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("Cannot compute a finite initial packet") from error


def potential(p: Parameters) -> np.ndarray:
    x = make_grid(p)
    return p.barrier_height * ((x >= p.barrier_left) & (x < p.barrier_right))
