from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tunneling.model import Parameters, initial_amplitudes
from tunneling.quantum import simulate_free
from tunneling.reference import simulate as simulate_reference


class QdkFreeEvolutionTests(unittest.TestCase):
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
