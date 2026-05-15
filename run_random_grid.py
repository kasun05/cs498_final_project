import subprocess
from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("results/random_grid")


def run_command(num_facilities: int, num_customers: int, seed: int, capacity_slack: float) -> None:
    cmd = [
        "python",
        "-m",
        "src.run_experiments",
        "--random",
        "--num-facilities",
        str(num_facilities),
        "--num-customers",
        str(num_customers),
        "--seed",
        str(seed),
        "--capacity-slack",
        str(capacity_slack),
        "--results-dir",
        str(RESULTS_DIR),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def combine_results() -> None:
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))

    if not csv_files:
        raise RuntimeError(f"No CSV files found in {RESULTS_DIR}")

    dfs = [pd.read_csv(path) for path in csv_files]
    combined = pd.concat(dfs, ignore_index=True)

    output_path = RESULTS_DIR / "combined_random_grid_results.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nCombined {len(csv_files)} files.")
    print(f"Saved combined results to: {output_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sizes = [
        (10, 30),
        (20, 60),
        (30, 100),
    ]

    seeds = range(5)
    capacity_slack = 1.2

    for num_facilities, num_customers in sizes:
        for seed in seeds:
            run_command(
                num_facilities=num_facilities,
                num_customers=num_customers,
                seed=seed,
                capacity_slack=capacity_slack,
            )

    combine_results()


if __name__ == "__main__":
    main()