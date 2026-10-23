"""Shared diagnostics; inside-barrier probability is reported separately."""
import numpy as np
from .model import Parameters, make_grid


def region_probabilities(alpha: np.ndarray, p: Parameters) -> dict[str, float]:
    alpha = np.asarray(alpha)
    if alpha.shape != (p.size,) or not np.all(np.isfinite(alpha)):
        raise ValueError("Expected a finite amplitude vector matching the grid")
    x = make_grid(p)
    probability = np.abs(alpha) ** 2
    return {
        "left": float(probability[x < p.barrier_left].sum()),
        "inside": float(probability[(x >= p.barrier_left) & (x < p.barrier_right)].sum()),
        "right": float(probability[x >= p.barrier_right].sum()),
        "norm": float(probability.sum()),
    }

# Only interpret left/right as R/T after the outgoing packets have separated,
# with negligible initial right tail and before boundary wraparound.
