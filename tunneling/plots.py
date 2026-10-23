"""Save figures without needing a GUI; files go to results/."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
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


def animate_packet(states: np.ndarray, p: Parameters, destination: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    try:
        x = make_grid(p)
        line, = ax.plot(x, np.abs(states[0]) ** 2 / p.dx, label="Probability density")
        ax.set(
            xlabel="x", ylabel="Probability density",
            xlim=(p.x_min, p.x_max),
            ylim=(0, 1.1 * np.max(np.abs(states)) ** 2 / p.dx),
        )
        title = ax.set_title("Wave packet: t = 0")
        other = ax.twinx()
        other.step(x, potential(p), where="post", color="tab:orange", alpha=0.6)
        other.set_ylabel("V(x)", color="tab:orange")
        other.set_ylim(0, 1.1 * p.barrier_height if p.barrier_height else 1)
        ax.legend(loc="upper left")
        frames = np.linspace(0, p.steps, min(p.steps + 1, 120), dtype=int)
        writer = PillowWriter(fps=20)
        with writer.saving(fig, str(destination), dpi=100):
            for index in frames:
                line.set_ydata(np.abs(states[index]) ** 2 / p.dx)
                title.set_text(f"Wave packet: t = {index * p.dt:.3g}")
                writer.grab_frame()
    finally:
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
