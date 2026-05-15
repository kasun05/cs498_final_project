import subprocess
import sys
from pathlib import Path

import pandas as pd


BENCHMARK_DIR = Path("benchmarks/orlib_cap")
RESULTS_DIR = Path("results/orlib")
COMBINED_OUTPUT = RESULTS_DIR / "combined_orlib_results.csv"


def expected_output_path(instance_path: Path) -> Path:
    return RESULTS_DIR / f"{instance_path.stem}_results.csv"


def run_instance(instance_path: Path, time_limit: int) -> None:
    output_path = expected_output_path(instance_path)

    if output_path.exists():
        print(f"Skipping existing result: {output_path}", flush=True)
        return

    cmd = [
        sys.executable,
        "-m",
        "src.run_experiments",
        "--instance-path",
        str(instance_path),
        "--time-limit",
        str(time_limit),
        "--results-dir",
        str(RESULTS_DIR),
    ]

    print("\nRunning:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def combine_results() -> None:
    csv_files = sorted(
        path for path in RESULTS_DIR.glob("*.csv")
        if path.name != COMBINED_OUTPUT.name
    )

    if not csv_files:
        raise RuntimeError(f"No OR-Library result CSVs found in {RESULTS_DIR}")

    dfs = [pd.read_csv(path) for path in csv_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(COMBINED_OUTPUT, index=False)

    print(f"\nCombined {len(csv_files)} OR-Library result files.")
    print(f"Saved combined results to: {COMBINED_OUTPUT}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    instance_paths = sorted(BENCHMARK_DIR.glob("cap*.txt"))

    if not instance_paths:
        raise RuntimeError(
            f"No cap*.txt files found in {BENCHMARK_DIR}. "
            "Run python download_orlib.py first."
        )

    # Keep this controlled. Your random 30x100 cases already took a while.
    time_limit = 300

    for instance_path in instance_paths:
        run_instance(instance_path, time_limit=time_limit)

    combine_results()


if __name__ == "__main__":
    main()