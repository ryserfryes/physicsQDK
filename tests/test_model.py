from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tunneling.analysis import region_probabilities
from tunneling.model import (
    Parameters, initial_amplitudes, load_parameters, make_grid, potential,
)


class InitialConditionsTests(unittest.TestCase):
    def test_numeric_parameters_accept_ints_and_floats(self):
        p = Parameters(mass=2, hbar=2.5, sigma_x=1, p0=-1, barrier_height=0)
        self.assertEqual((p.mass, p.hbar, p.sigma_x, p.p0), (2, 2.5, 1, -1))
        self.assertEqual(p.barrier_height, 0)

    def test_invalid_types_and_nonfinite_values_name_the_field(self):
        cases = (
            ("mass", True), ("barrier_height", False), ("sigma_x", "1.5"),
            ("sigma_x", [1.5]), ("p0", np.array([1.0, 2.0])),
            ("x0", np.nan), ("hbar", np.inf), ("mass", -np.inf),
            ("n_qubits", 8.0), ("steps", True), ("shots", "2000"),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    Parameters(**{field: value})

    def test_invalid_ranges_are_rejected(self):
        cases = (
            {"n_qubits": 0}, {"n_qubits": 13}, {"x_min": 20}, {"x0": -20},
            {"sigma_x": 0}, {"mass": 0}, {"hbar": 0}, {"dt": 0},
            {"barrier_width": 0}, {"barrier_height": -1},
            {"steps": 0}, {"shots": 0}, {"barrier_center": 20},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Parameters(**values)

    def test_toml_overrides_defaults_and_validates_values(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.toml"
            path.write_text("mass = 2\np0 = -1.25\n", encoding="utf-8")
            self.assertEqual(load_parameters(path), replace(Parameters(), mass=2, p0=-1.25))
            path.write_text('sigma_x = "1.5"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sigma_x"):
                load_parameters(path)

    def test_grid_and_barrier_use_half_open_intervals(self):
        p = Parameters(n_qubits=3, x_min=-2, x_max=2, x0=-1, barrier_height=3)
        np.testing.assert_array_equal(make_grid(p), [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        self.assertEqual((p.size, p.dx), (8, 0.5))
        np.testing.assert_array_equal(potential(p), [0, 0, 0, 3, 3, 0, 0, 0])
        np.testing.assert_array_equal(potential(replace(p, barrier_height=0)), np.zeros(8))

    def test_packet_is_normalized_with_expected_position_moments(self):
        p = Parameters(n_qubits=10, x_min=-40, x_max=40, x0=-8, sigma_x=1.5)
        x, alpha = make_grid(p), initial_amplitudes(p)
        self.assertEqual(alpha.shape, (p.size,))
        self.assertTrue(np.iscomplexobj(alpha))
        self.assertTrue(np.all(np.isfinite(alpha)))
        probability = np.abs(alpha) ** 2
        mean = float(x @ probability)
        variance = float((x - mean) ** 2 @ probability)
        self.assertAlmostEqual(float(probability.sum()), 1, delta=1e-12)
        self.assertAlmostEqual(mean, p.x0, delta=1e-10)
        self.assertAlmostEqual(variance, p.sigma_x ** 2, delta=1e-10)

    def test_packet_phase_tracks_momentum_and_hbar(self):
        for momentum in (0, 1.2, -1.2):
            with self.subTest(momentum=momentum):
                p = Parameters(p0=momentum, hbar=2)
                alpha = initial_amplitudes(p)
                phase_steps = np.angle(alpha[1:] * alpha[:-1].conj())
                np.testing.assert_allclose(phase_steps, momentum * p.dx / p.hbar, atol=1e-12)

    def test_unrepresentable_packet_fails_but_small_tails_are_allowed(self):
        for width in (1e-200, 1e200):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "packet"):
                initial_amplitudes(Parameters(sigma_x=width))
        alpha = initial_amplitudes(Parameters(sigma_x=0.05))
        self.assertTrue(np.all(np.isfinite(alpha)))
        self.assertTrue(np.any(alpha == 0))
        self.assertAlmostEqual(float(np.sum(np.abs(alpha) ** 2)), 1, delta=1e-12)

    def test_region_probabilities_partition_the_state(self):
        p = Parameters(n_qubits=3, x_min=-2, x_max=2, x0=-1)
        alpha = np.sqrt([0.05, 0.1, 0.15, 0.2, 0.25, 0.15, 0.05, 0.05])
        result = region_probabilities(alpha, p)
        for region, expected in (("left", 0.3), ("inside", 0.45), ("right", 0.25)):
            self.assertAlmostEqual(result[region], expected, delta=1e-12)
        self.assertAlmostEqual(result["norm"], 1, delta=1e-12)
        self.assertAlmostEqual(sum(result[key] for key in ("left", "inside", "right")), result["norm"])
