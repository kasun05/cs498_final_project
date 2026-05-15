from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data_utils import generate_random_instance, load_orlib_cap_instance
from .metrics import solve_and_measure
from .models import FORMULATION_BUILDERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run facility-location formulation comparison experiments.")
    parser.add_argument("--random", action="store_true", help="Generate a random instance instead of loading from file.")
    parser.add_argument("--num-facilities", type=int, default=10)
    parser.add_argument("--num-customers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instance-path", type=str, default=None, help="Path to an OR-Library-style cap instance.")
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--capacity-slack", type=float, default=1.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.random:
        instance = generate_random_instance(
            num_facilities=args.num_facilities,
            num_customers=args.num_customers,
            seed=args.seed,
            capacity_slack=args.capacity_slack,
        )
    elif args.instance_path:
        instance = load_orlib_cap_instance(args.instance_path)
    else:
        raise ValueError("Provide either --random or --instance-path.")

    rows = []
    for formulation_name, builder in FORMULATION_BUILDERS.items():
        result = solve_and_measure(
            instance=instance,
            formulation_name=formulation_name,
            model_builder=builder,
            time_limit_sec=args.time_limit,
        )
        rows.append(result.to_dict())

    df = pd.DataFrame(rows)
    df["seed"] = args.seed if args.random else None
    df["instance_source"] = "random" if args.random else "or_library"
    df["capacity_slack_param"] = args.capacity_slack if args.random else None

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.random:
        slack_str = str(args.capacity_slack).replace(".", "p")
        output_name = f"{instance.name}_slack{slack_str}_results.csv"
    else:
        output_name = f"{instance.name}_results.csv"

    output_path = results_dir / output_name
    df.to_csv(output_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()