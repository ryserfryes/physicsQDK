import builtins
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tunneling.model import initial_amplitudes, load_parameters, make_grid


class PacketCliTests(unittest.TestCase):
    def test_packet_writes_readable_outputs_without_importing_qdk(self):
        config = Path(__file__).resolve().parents[1] / "config" / "default.toml"
        original_import = builtins.__import__

        def without_qdk(name, *args, **kwargs):
            if name.split(".")[0] in ("qdk", "qsharp"):
                self.fail(f"Packet preview must not import {name}")
            return original_import(name, *args, **kwargs)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache_env = {"MPLCONFIGDIR": str(root / "matplotlib"), "XDG_CACHE_HOME": str(root)}
            with patch.dict(os.environ, cache_env):
                with patch("builtins.__import__", new=without_qdk):
                    import run
                    from matplotlib.image import imread

                    args = ["run.py", "packet", "--config", str(config)]
                    with patch.object(run, "ROOT", root), patch.object(sys, "argv", args):
                        with redirect_stdout(io.StringIO()):
                            run.main()

                    p = load_parameters(config)
                    with np.load(root / "results" / "initial_packet.npz", allow_pickle=False) as data:
                        self.assertEqual(set(data.files), {"x", "alpha", "dx"})
                        np.testing.assert_array_equal(data["x"], make_grid(p))
                        np.testing.assert_array_equal(data["alpha"], initial_amplitudes(p))
                        self.assertEqual(data["dx"].item(), p.dx)
                    image = imread(root / "results" / "initial_packet.png")
                    self.assertGreater(image.shape[0], 0)
                    self.assertGreater(image.shape[1], 0)
                    self.assertTrue(np.all(np.isfinite(image)))
