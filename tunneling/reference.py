import numpy as np

from .model import Parameters, initial_amplitudes


def simulate(p: Parameters) -> np.ndarray:
    if p.barrier_height != 0:
        raise ValueError("Free evolution requires barrier_height = 0")

    try:
        with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
            scaled_energy = (
                p.dt * p.hbar / (p.mass * np.square(p.dx))
                * np.sin(np.pi * np.fft.fftfreq(p.size))**2
            )
            step = (1 - 1j * scaled_energy) / (1 + 1j * scaled_energy)
            states = np.empty((p.steps + 1, p.size), dtype=np.complex128)
            states[0] = initial_amplitudes(p)
            spectrum = np.fft.fft(states[0])
            for i in range(1, p.steps + 1):
                spectrum *= step
                states[i] = np.fft.ifft(spectrum)
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise ValueError("Cannot compute finite free evolution for these parameters") from error

    if not np.all(np.isfinite(states)):
        raise ValueError("Free evolution produced nonfinite amplitudes")
    return states
