#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def column(rows, key):
    return [r[key] for r in rows if key in r]


def main():
    parser = argparse.ArgumentParser(description="Plot ILQR solve output CSV files.")
    parser.add_argument("--prefix", required=True, help="CSV prefix used by ILQR::save_solve_output_csv")
    parser.add_argument("--save", default="", help="Optional output PNG path")
    args = parser.parse_args()

    prefix = Path(args.prefix)
    ref_states = read_csv(Path(f"{prefix}_reference_states.csv"))
    init_states = read_csv(Path(f"{prefix}_initial_states.csv"))
    final_states = read_csv(Path(f"{prefix}_final_states.csv"))
    init_controls = read_csv(Path(f"{prefix}_initial_controls.csv"))
    final_controls = read_csv(Path(f"{prefix}_final_controls.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0][0]
    if ref_states:
        ax.plot(column(ref_states, "x0"), column(ref_states, "x1"), "k--", label="reference")
    if init_states:
        ax.plot(column(init_states, "x0"), column(init_states, "x1"), "r-", label="initial")
    if final_states:
        ax.plot(column(final_states, "x0"), column(final_states, "x1"), "b-", label="final")
    ax.set_title("XY Trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')

    ax = axes[0][1]
    if ref_states and "x2" in ref_states[0]:
        ax.plot(column(ref_states, "index"), column(ref_states, "x2"), "k--", label="ref yaw")
    if init_states and "x2" in init_states[0]:
        ax.plot(column(init_states, "index"), column(init_states, "x2"), "r-", label="init yaw")
    if final_states and "x2" in final_states[0]:
        ax.plot(column(final_states, "index"), column(final_states, "x2"), "b-", label="final yaw")
    if ref_states and "x3" in ref_states[0]:
        ax.plot(column(ref_states, "index"), column(ref_states, "x3"), "k:", label="ref speed")
    if init_states and "x3" in init_states[0]:
        ax.plot(column(init_states, "index"), column(init_states, "x3"), "r:", label="init speed")
    if final_states and "x3" in final_states[0]:
        ax.plot(column(final_states, "index"), column(final_states, "x3"), "b:", label="final speed")
    ax.set_title("Yaw and Speed")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True)

    ax = axes[1][0]
    if init_controls and "u0" in init_controls[0]:
        ax.plot(column(init_controls, "index"), column(init_controls, "u0"), "r-", label="init accel")
    if final_controls and "u0" in final_controls[0]:
        ax.plot(column(final_controls, "index"), column(final_controls, "u0"), "b-", label="final accel")
    ax.set_title("Acceleration Control")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True)

    ax = axes[1][1]
    if init_controls and "u1" in init_controls[0]:
        ax.plot(column(init_controls, "index"), column(init_controls, "u1"), "r-", label="init steer")
    if final_controls and "u1" in final_controls[0]:
        ax.plot(column(final_controls, "index"), column(final_controls, "u1"), "b-", label="final steer")
    ax.set_title("Steering Control")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
