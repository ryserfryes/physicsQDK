import argparse
import json
from pathlib import Path
import numpy as np
from tunneling.model import load_parameters, make_grid, initial_amplitudes
from tunneling.analysis import region_probabilities
from tunneling.plots import plot_packet, plot_register

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Quantum tunneling project scaffold")
    parser.add_argument("mode", choices=["register", "packet"])
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.toml")
    parser.add_argument("--qubits", type=int, default=3, help="Register demo only")
    parser.add_argument("--shots", type=int, default=None, help="Register demo only")
    args = parser.parse_args()
    p = load_parameters(args.config)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    if args.mode == "register":
        from tunneling.quantum import sample_register
        samples = sample_register(ROOT, args.qubits, p.shots if args.shots is None else args.shots)
        np.savetxt(output / "register_samples.csv", samples, fmt="%d", header="position_index", comments="")
        plot_register(samples, args.qubits, output / "register.png")
        print(f"QDK register OK. Samples: {len(samples)}; points: {2 ** args.qubits}")
        print(f"Figure: {output / 'register.png'}")
    else:
        alpha = initial_amplitudes(p)
        np.savez(output / "initial_packet.npz", x=make_grid(p), alpha=alpha, dx=p.dx)
        plot_packet(alpha, p, output / "initial_packet.png")
        print("Initial conditions only; Q# state preparation and evolution are TODO.")
        print(json.dumps(region_probabilities(alpha, p), indent=2))
        print(f"Figure: {output / 'initial_packet.png'}")


if __name__ == "__main__":
    main()
