from dataclasses import replace
import unittest

import numpy as np

from tunneling.model import Parameters, initial_amplitudes, make_grid
from tunneling.reference import simulate


def hamiltonian(p):
    identity = np.eye(p.size)
    laplacian = np.roll(identity, 1, axis=0) - 2 * identity + np.roll(identity, -1, axis=0)
    return -p.hbar ** 2 * laplacian / (2 * p.mass * p.dx ** 2)


class ReferenceTests(unittest.TestCase):
    def test_history_includes_initial_state_and_preserves_norm(self):
        p = Parameters(n_qubits=7, barrier_height=0, steps=80)
        states = simulate(p)
        self.assertEqual(states.shape, (p.steps + 1, p.size))
        self.assertTrue(np.iscomplexobj(states))
        self.assertTrue(np.all(np.isfinite(states)))
        np.testing.assert_array_equal(states[0], initial_amplitudes(p))
        np.testing.assert_allclose(np.sum(np.abs(states) ** 2, axis=1), 1, atol=1e-10, rtol=0)

    def test_matches_repeated_dense_crank_nicolson_solves(self):
        for qubits in (1, 3, 5):
            with self.subTest(size=2 ** qubits):
                p = Parameters(n_qubits=qubits, x_min=-4, x_max=4, x0=-1.5,
                               sigma_x=0.7, p0=0.8, mass=0.7, hbar=1.3,
                               barrier_height=0, dt=0.04, steps=7)
                identity = np.eye(p.size)
                scaled = 0.5j * p.dt * hamiltonian(p) / p.hbar
                expected = [initial_amplitudes(p)]
                for _ in range(p.steps):
                    expected.append(np.linalg.solve(identity + scaled, (identity - scaled) @ expected[-1]))
                np.testing.assert_allclose(simulate(p), expected, atol=2e-12, rtol=0)

    def test_free_packet_position_and_spreading_for_both_directions(self):
        for momentum in (0, 1, -1):
            with self.subTest(momentum=momentum):
                p = Parameters(n_qubits=10, x_min=-20, x_max=20, x0=-8,
                               sigma_x=1.5, p0=momentum, mass=1, hbar=1,
                               barrier_height=0, dt=0.01, steps=100)
                x = make_grid(p)
                probability = np.abs(simulate(p)) ** 2
                means = probability @ x
                variances = np.sum(probability * (x[None, :] - means[:, None]) ** 2, axis=1)
                times = p.dt * np.arange(p.steps + 1)
                expected_means = p.x0 + p.p0 * times / p.mass
                expected_variances = p.sigma_x ** 2 + (p.hbar * times / (2 * p.mass * p.sigma_x)) ** 2
                np.testing.assert_allclose(means, expected_means, atol=5e-4, rtol=0)
                np.testing.assert_allclose(variances, expected_variances, atol=3e-4, rtol=0)

    def test_time_error_is_second_order_against_exact_discrete_evolution(self):
        p = Parameters(n_qubits=5, barrier_height=0)
        energies, vectors = np.linalg.eigh(hamiltonian(p))
        exact = vectors @ (np.exp(-1j * energies / p.hbar) * (vectors.conj().T @ initial_amplitudes(p)))
        errors = []
        for dt in (0.2, 0.1, 0.05):
            final = simulate(replace(p, dt=dt, steps=round(1 / dt)))[-1]
            errors.append(np.linalg.norm(final - exact))
        np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 4, atol=0.2, rtol=0)

    def test_spatial_error_is_second_order_against_analytic_density(self):
        errors = []
        for qubits in (7, 8, 9):
            p = Parameters(n_qubits=qubits, x_min=-20, x_max=20, x0=-8,
                           sigma_x=1.5, p0=1, mass=1, hbar=1,
                           barrier_height=0, dt=0.001, steps=1000)
            time = p.dt * p.steps
            mean = p.x0 + p.p0 * time / p.mass
            variance = p.sigma_x ** 2 + (p.hbar * time / (2 * p.mass * p.sigma_x)) ** 2
            exact = np.exp(-(make_grid(p) - mean) ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
            density = np.abs(simulate(p)[-1]) ** 2 / p.dx
            errors.append(np.sqrt(p.dx * np.sum((density - exact) ** 2)))
        np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 4, atol=0.2, rtol=0)

    def test_nonzero_barrier_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate(Parameters(barrier_height=1))

    def test_arithmetic_breakdown_raises_value_error(self):
        for values in ({"dt": 1e308}, {"mass": 5e-324}, {"n_qubits": 1, "dt": 1e308, "mass": 1e308}):
            with self.subTest(values=values):
                p = Parameters(barrier_height=0, p0=0, steps=1, **values)
                self.assertTrue(np.all(np.isfinite(initial_amplitudes(p))))
                with self.assertRaises(ValueError):
                    simulate(p)
