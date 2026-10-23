"""Save figures without needing a GUI; files go to results/."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .model import Parameters, make_grid, potential


def plot_packet(alpha: np.ndarray, p: Parameters, destination: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.plot(make_grid(p), np.abs(alpha) ** 2 / p.dx, label="Initial probability density")
    ax.set(xlabel="x", ylabel="Probability density", title="Gaussian packet: initial conditions only")
    other = ax.twinx()
    other.step(make_grid(p), potential(p), where="post", color="tab:orange", alpha=0.6)
    other.set_ylabel("V(x)", color="tab:orange")
    ax.legend(loc="upper left")
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_register(samples: np.ndarray, n_qubits: int, destination: Path):
    size = 2 ** n_qubits
    counts = np.bincount(samples, minlength=size)
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.bar(np.arange(size), counts / len(samples), label="QDK measurements")
    ax.axhline(1 / size, color="tab:orange", label="Expected probability")
    ax.set(xlabel="Position index j", ylabel="Probability", title=f"Coordinate register: {n_qubits} qubits, {len(samples)} shots")
    ax.legend()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
