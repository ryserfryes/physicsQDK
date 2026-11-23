from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tunneling.model import Parameters, initial_amplitudes
from tunneling.quantum import simulate, simulate_free
from tunneling.reference import simulate as simulate_reference


class QdkFreeEvolutionTests(unittest.TestCase):
    def test_second_barrier_has_effect_and_conserves_probability(self):
        p = Parameters(n_qubits=7, x0=-2, sigma_x=1, p0=1, steps=20, dt=0.05)
        one = simulate(p, [20], ROOT)[0]
        two = simulate(p, [20], ROOT, second_barrier_center=2)[0]
        self.assertGreater(np.linalg.norm(one - two), 1e-4)
        self.assertAlmostEqual(float(np.sum(abs(two) ** 2)), 1, delta=1e-9)
        with self.assertRaises(ValueError):
            simulate(p, [1], ROOT, second_barrier_center=0.5)

    def test_barrier_changes_evolution_and_matches_reference(self):
        p = Parameters(n_qubits=8, x0=-2, sigma_x=1, p0=1, steps=20, dt=0.05)
        state = simulate(p, [20], ROOT)[0]
        free = simulate_free(p, [20], ROOT)[0]
        reference = simulate_reference(p)[-1]
        self.assertGreater(np.linalg.norm(state - free), 0.01)
        self.assertAlmostEqual(float(np.sum(np.abs(state) ** 2)), 1, delta=1e-9)
        np.testing.assert_allclose(np.abs(state) ** 2, np.abs(reference) ** 2, atol=0.002, rtol=0)

    def test_packet_and_free_evolution_match_python_at_multiple_times(self):
        p = Parameters(n_qubits=7, x_min=-20, x_max=20, x0=-8,
                       sigma_x=2, p0=1, barrier_height=0, dt=0.02, steps=10)
        steps = [0, 5, 10]
        qdk_states = simulate_free(p, steps, ROOT)
        reference = simulate_reference(replace(p, steps=steps[-1]))[steps]
        self.assertEqual(qdk_states.shape, (3, p.size))
        np.testing.assert_allclose(np.sum(np.abs(qdk_states) ** 2, axis=1), 1, atol=1e-10)
        np.testing.assert_allclose(np.abs(qdk_states[0]), np.abs(initial_amplitudes(p)), atol=1e-10)
        for qdk_state, reference_state in zip(qdk_states, reference):
            overlap = np.vdot(reference_state, qdk_state)
            aligned = qdk_state * overlap.conjugate() / abs(overlap)
            self.assertLess(np.linalg.norm(aligned - reference_state), 0.003)


if __name__ == "__main__":
    unittest.main()
