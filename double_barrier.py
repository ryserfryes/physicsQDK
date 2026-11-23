"""Scan incident momenta for two identical barriers on the QDK simulator."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np

from tunneling.model import load_parameters, make_grid
from tunneling.quantum import simulate

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.toml")
    parser.add_argument("--second-center", type=float, required=True)
    parser.add_argument("--momenta", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/double_barrier.npz")
    args = parser.parse_args()
    p = load_parameters(args.config)
    x = make_grid(p)
    left = p.barrier_left
    right = args.second_center + p.barrier_width / 2
    edge_width = (p.x_max - p.x_min) * 0.05
    rows, states = [], []
    for momentum in args.momenta:
        current = replace(p, p0=momentum)
        initial, final = simulate(current, [0, p.steps], ROOT, second_barrier_center=args.second_center)
        probability = abs(final) ** 2
        row = [float(probability[x < left].sum()), float(probability[(x >= left) & (x < right)].sum()),
               float(probability[x >= right].sum()),
               float(probability[(x < p.x_min + edge_width) | (x >= p.x_max - edge_width)].sum()),
               float((abs(initial[x >= left]) ** 2).sum())]
        rows.append(row)
        states.append(final)
        print(f"p={momentum:g} E={momentum**2/(2*p.mass):g} left={row[0]:.6f} between={row[1]:.6f} right={row[2]:.6f} edges={row[3]:.6f} initial_past_first={row[4]:.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, x=x, momenta=args.momenta, probabilities=rows, states=states,
             columns=['left', 'between_outer_edges', 'right', 'edges', 'initial_past_first'],
             parameters=json.dumps(asdict(p)), second_barrier_center=args.second_center)
    print(f"Saved: {args.output}; interpret right as transmission only after separation and before boundary wraparound.")


if __name__ == "__main__":
    main()
