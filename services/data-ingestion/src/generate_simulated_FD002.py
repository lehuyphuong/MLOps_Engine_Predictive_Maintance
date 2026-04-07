"""
Generates simulated_FD002.txt — a two-stage dataset for live stream simulation:

  Stage 1 — Normal (50 synthetic cycles per engine)
    Sensors are stable around the healthy baseline of that engine.
    Small Gaussian noise is added to mimic real sensor jitter.
    TimeInCycles counts from 1 upward.
    RUL = 50 + original_max_cycles (engine has full life ahead)

  Stage 2 — Degradation (original train_FD002.txt rows)
    Real degradation trajectories from C-MAPSS.
    TimeInCycles continues from where Stage 1 left off.
    RUL = max_cycle - current_cycle (standard C-MAPSS calculation)

Command:
    python generate_simulated_FD002.py \
        --data-dir /path/to/CMaps \
        --output-dir /path/to/CMaps \
        --normal-cycles 50 \
        --noise-std 0.01
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Column definitions — must match existing preprocessing script exactly
# ---------------------------------------------------------------------------
COLUMN_NAMES = (
    ["UnitNumber", "TimeInCycles"]
    + [f"OperSet{i}"   for i in range(1, 4)]
    + [f"SensorMes{j}" for j in range(1, 22)]
)

SENSOR_COLS = [f"SensorMes{j}" for j in range(1, 22)]
OPSET_COLS  = [f"OperSet{i}"   for i in range(1, 4)]


def load_train(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "train_FD002.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    print(f"Loaded {len(df)} rows, {df['UnitNumber'].nunique()} engines from {path}")
    return df


def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add RUL column — matches rul_train_generation() in your existing script."""
    max_cycles = df.groupby("UnitNumber")["TimeInCycles"].max().rename("max_cycle")
    df = df.join(max_cycles, on="UnitNumber")
    df["RUL"] = df["max_cycle"] - df["TimeInCycles"]
    df.drop(columns="max_cycle", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Stage 1 — generate synthetic normal cycles
# ---------------------------------------------------------------------------
def generate_normal_stage(
    df: pd.DataFrame,
    normal_cycles: int,
    noise_std: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    For each engine, take the very first cycle as the healthy baseline.
    Replicate it `normal_cycles` times with small Gaussian noise on sensors.
    Operational settings are kept identical (they represent flight conditions,
    not health state, so they should not drift).

    TimeInCycles: 1 .. normal_cycles
    RUL: normal_cycles + max_degradation_cycles - current_cycle
         (engine has full degradation life still ahead of it)
    """
    normal_rows = []

    for unit_id, group in df.groupby("UnitNumber"):
        # Healthy baseline = first recorded cycle of this engine
        baseline = group.sort_values("TimeInCycles").iloc[0]

        # How many real degradation cycles does this engine have?
        max_degradation_cycles = group["TimeInCycles"].max()

        for cycle in range(1, normal_cycles + 1):
            row = {}
            row["UnitNumber"]   = unit_id
            row["TimeInCycles"] = cycle

            # Operational settings — preserve exact values (no noise)
            for col in OPSET_COLS:
                row[col] = baseline[col]

            # Sensors — baseline + small Gaussian jitter
            for col in SENSOR_COLS:
                noise    = rng.normal(0, noise_std * abs(baseline[col]) + 1e-6)
                row[col] = round(float(baseline[col]) + noise, 4)

            # RUL: cycles remaining = normal window left + full degradation ahead
            row["RUL"] = (normal_cycles - cycle) + max_degradation_cycles

            normal_rows.append(row)

    normal_df = pd.DataFrame(normal_rows, columns=COLUMN_NAMES + ["RUL"])
    print(
        f"Stage 1 — generated {len(normal_df)} normal cycles "
        f"({normal_cycles} per engine x {df['UnitNumber'].nunique()} engines)"
    )
    return normal_df


# ---------------------------------------------------------------------------
# Stage 2 — real degradation cycles with re-indexed TimeInCycles
# ---------------------------------------------------------------------------
def build_degradation_stage(
    df: pd.DataFrame,
    normal_cycles: int,
) -> pd.DataFrame:
    """
    Take original train_FD002 rows and shift TimeInCycles forward by
    `normal_cycles` so the timeline is continuous after Stage 1.

    RUL is recalculated from the shifted cycle index so it stays consistent.
    """
    df = df.copy()
    df["TimeInCycles"] = df["TimeInCycles"] + normal_cycles

    # Recalculate RUL from shifted cycle (RUL should still reach 0 at failure)
    max_shifted = df.groupby("UnitNumber")["TimeInCycles"].max().rename("max_shifted")
    df = df.join(max_shifted, on="UnitNumber")
    df["RUL"] = df["max_shifted"] - df["TimeInCycles"]
    df.drop(columns="max_shifted", inplace=True)

    print(
        f"Stage 2 — {len(df)} real degradation cycles "
        f"(TimeInCycles shifted by +{normal_cycles})"
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate simulated_FD002.txt")
    parser.add_argument("--data-dir",      default="data/CMaps",     help="Folder with train_FD002.txt")
    parser.add_argument("--output-dir",    default="data/CMaps",     help="Where to write simulated_FD002.txt")
    parser.add_argument("--normal-cycles", type=int,   default=50,   help="Normal cycles to prepend per engine")
    parser.add_argument("--noise-std",     type=float, default=0.01, help="Gaussian noise std (fraction of value)")
    parser.add_argument("--seed",          type=int,   default=42,   help="Random seed for reproducibility")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Load and compute RUL on real data
    train_df = load_train(data_dir)
    train_df = compute_rul(train_df)

    # Build both stages
    stage1 = generate_normal_stage(train_df, args.normal_cycles, args.noise_std, rng)
    stage2 = build_degradation_stage(train_df, args.normal_cycles)

    # Combine — Stage 1 first, then Stage 2, sorted by engine then cycle
    combined = pd.concat([stage1, stage2], ignore_index=True)
    combined = combined.sort_values(["UnitNumber", "TimeInCycles"]).reset_index(drop=True)

    # Drop RUL — output format matches original .txt (no RUL column)
    # RUL is computed at training time by rul_train_generation() in your script
    output_cols  = COLUMN_NAMES
    combined_out = combined[output_cols]

    # Write space-separated, no header, no index — identical format to original files
    output_path = output_dir / "simulated_FD002.txt"
    combined_out.to_csv(output_path, sep=" ", header=False, index=False)

    print(f"\nOutput written to  : {output_path}")
    print(f"Total rows         : {len(combined_out)}")
    print(f"Total engines      : {combined_out['UnitNumber'].nunique()}")
    print(f"Cycles range       : {combined_out['TimeInCycles'].min()} -> {combined_out['TimeInCycles'].max()}")
    print(f"\nStage breakdown per engine:")
    print(f"  Stage 1 (normal)      : cycles 1 -> {args.normal_cycles}")
    print(f"  Stage 2 (degradation) : cycles {args.normal_cycles + 1} -> end")


if __name__ == "__main__":
    main()