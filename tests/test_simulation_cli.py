import builtins
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tunneling.model import Parameters, initial_amplitudes, load_parameters, make_grid
from tunneling.reference import simulate


class SimulationCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cache = Path(cls.enterClassContext(TemporaryDirectory()))
        cls.enterClassContext(patch.dict(os.environ, {
            "MPLCONFIGDIR": str(cache / "matplotlib"), "XDG_CACHE_HOME": str(cache),
        }))

    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        original_import = builtins.__import__

        def without_qdk(name, *args, **kwargs):
            if name.split(".")[0] in ("qdk", "qsharp"):
                self.fail(f"Simulation must not import {name}")
            return original_import(name, *args, **kwargs)

        self.enterContext(patch("builtins.__import__", new=without_qdk))
        import run

        self.run = run
        self.config = self.root / "parameters.toml"
        self.enterContext(patch.object(run, "ROOT", self.root))
        self.enterContext(patch.object(sys, "argv", ["run.py", "simulate", "--config", str(self.config)]))

    def test_simulation_writes_complete_history_and_animated_gif_without_qdk(self):
        self.config.write_text(
            "n_qubits = 6\nx0 = -4.0\nsigma_x = 0.8\np0 = 1.2\ndt = 0.2\nsteps = 4\n",
            encoding="utf-8",
        )
        with redirect_stdout(io.StringIO()):
            self.run.main()
        p = load_parameters(self.config)
        output = self.root / "results"
        with np.load(output / "simulation.npz", allow_pickle=False) as data:
            self.assertEqual(set(data.files), {"x", "times", "states", "dx", "parameters"})
            self.assertEqual(data["states"].shape, (5, 64))
            np.testing.assert_array_equal(data["x"], make_grid(p))
            np.testing.assert_array_equal(data["times"], p.dt * np.arange(p.steps + 1))
            np.testing.assert_array_equal(data["states"][0], initial_amplitudes(p))
            np.testing.assert_array_equal(data["states"], simulate(p))
            self.assertEqual(data["dx"].item(), p.dx)
            parameters = json.loads(data["parameters"].item())
            self.assertEqual(parameters, asdict(p))
            self.assertEqual(Parameters(**parameters), p)
        with Image.open(output / "simulation.gif") as gif:
            self.assertEqual(gif.format, "GIF")
            self.assertEqual(gif.size, (900, 450))
            self.assertTrue(gif.is_animated)
            self.assertEqual(gif.n_frames, p.steps + 1)
            first = np.array(gif.convert("RGB"))
            for index in range(gif.n_frames):
                gif.seek(index)
                self.assertEqual(gif.info["duration"], 50)
            last = np.array(gif.convert("RGB"))
            self.assertFalse(np.array_equal(first, last))

    def test_failed_calculation_does_not_write_results(self):
        self.config.write_text(
            "n_qubits = 3\nx_min = -2.0\nx_max = 2.0\nx0 = -1.0\n"
            "barrier_center = 0.25\nbarrier_width = 0.1\nbarrier_height = 1.0\nsteps = 1\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
            self.run.main()
        self.assertFalse((self.root / "results" / "simulation.npz").exists())
        self.assertFalse((self.root / "results" / "simulation.gif").exists())

    def test_animation_closes_figure_when_gif_write_fails(self):
        import matplotlib.pyplot as plt
        from tunneling.plots import animate_packet

        p = Parameters(n_qubits=4, steps=1)
        states = simulate(p)
        figures = set(plt.get_fignums())
        with self.assertRaises(OSError):
            animate_packet(states, p, self.root / "missing" / "simulation.gif")
        self.assertEqual(set(plt.get_fignums()), figures)
