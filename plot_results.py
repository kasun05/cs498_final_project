from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RANDOM_RESULTS_PATH = Path("results/random_grid/combined_random_grid_results.csv")
ORLIB_RESULTS_PATH = Path("results/orlib/combined_orlib_results.csv")
FIGURE_DIR = Path("figures")


FORMULATION_ORDER = ["loose_big_m", "tight_big_m", "indicator"]


def add_size_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["size_label"] = (
        df["num_facilities"].astype(int).astype(str)
        + " facilities, "
        + df["num_customers"].astype(int).astype(str)
        + " customers"
    )
    return df


def summarize_random(df: pd.DataFrame) -> pd.DataFrame:
    df = add_size_label(df)

    summary = (
        df.groupby(["size_label", "formulation"], as_index=False)
        .agg(
            mean_lp_gap=("lp_gap_fraction", "mean"),
            mean_runtime=("runtime_sec", "mean"),
            mean_nodes=("mip_nodes", "mean"),
            mean_cuts=("cuts_applied", "mean"),
            mean_mip_gap=("mip_gap", "mean"),
            num_runs=("formulation", "count"),
        )
    )

    summary["formulation"] = pd.Categorical(
        summary["formulation"],
        categories=FORMULATION_ORDER,
        ordered=True,
    )
    summary = summary.sort_values(["size_label", "formulation"])
    return summary


def summarize_orlib(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["instance_name", "formulation"], as_index=False)
        .agg(
            status=("status", "first"),
            objective_value=("objective_value", "first"),
            lp_gap_fraction=("lp_gap_fraction", "first"),
            runtime_sec=("runtime_sec", "first"),
            mip_nodes=("mip_nodes", "first"),
            cuts_applied=("cuts_applied", "first"),
            mip_gap=("mip_gap", "first"),
            num_facilities=("num_facilities", "first"),
            num_customers=("num_customers", "first"),
            capacity_slack=("capacity_slack", "first"),
        )
    )

    summary["formulation"] = pd.Categorical(
        summary["formulation"],
        categories=FORMULATION_ORDER,
        ordered=True,
    )

    return summary.sort_values(["instance_name", "formulation"])


def summarize_orlib_feasible(df: pd.DataFrame) -> pd.DataFrame:
    feasible = df[df["status"] == "OPTIMAL"].copy()

    summary = (
        feasible.groupby("formulation", as_index=False)
        .agg(
            mean_lp_gap=("lp_gap_fraction", "mean"),
            mean_runtime=("runtime_sec", "mean"),
            mean_nodes=("mip_nodes", "mean"),
            mean_cuts=("cuts_applied", "mean"),
            mean_mip_gap=("mip_gap", "mean"),
            num_runs=("formulation", "count"),
        )
    )

    summary["formulation"] = pd.Categorical(
        summary["formulation"],
        categories=FORMULATION_ORDER,
        ordered=True,
    )

    return summary.sort_values("formulation")


def plot_grouped_bar(
    summary: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    log_scale: bool = False,
    percent: bool = False,
) -> None:
    pivot = summary.pivot(
        index="size_label",
        columns="formulation",
        values=value_col,
    )

    pivot = pivot[FORMULATION_ORDER]

    if percent:
        pivot = pivot * 100.0

    ax = pivot.plot(kind="bar", figsize=(10, 6))

    ax.set_title(title)
    ax.set_xlabel("Instance size")
    ax.set_ylabel(ylabel)
    ax.legend(title="Formulation")
    ax.tick_params(axis="x", rotation=20)

    if log_scale:
        ax.set_yscale("log")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_orlib_metric(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    log_scale: bool = False,
    percent: bool = False,
    only_optimal: bool = True,
) -> None:
    df = df.copy()

    if only_optimal:
        df = df[df["status"] == "OPTIMAL"].copy()

    if df.empty:
        print(f"Skipping {output_path}; no rows to plot.")
        return

    df["formulation"] = pd.Categorical(
        df["formulation"],
        categories=FORMULATION_ORDER,
        ordered=True,
    )
    df = df.sort_values(["instance_name", "formulation"])

    pivot = df.pivot(
        index="instance_name",
        columns="formulation",
        values=value_col,
    )

    pivot = pivot[FORMULATION_ORDER]

    if percent:
        pivot = pivot * 100.0

    ax = pivot.plot(kind="bar", figsize=(11, 6))

    ax.set_title(title)
    ax.set_xlabel("OR-Library instance")
    ax.set_ylabel(ylabel)
    ax.legend(title="Formulation")
    ax.tick_params(axis="x", rotation=35)

    if log_scale:
        ax.set_yscale("log")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def check_objective_consistency(df: pd.DataFrame, label: str) -> None:
    feasible = df[df["objective_value"].notna()].copy()

    if feasible.empty:
        print(f"\n[{label}] No feasible objective values to compare.")
        return

    grouped = (
        feasible.groupby("instance_name")["objective_value"]
        .agg(["min", "max"])
        .reset_index()
    )
    grouped["objective_spread"] = grouped["max"] - grouped["min"]

    max_spread = grouped["objective_spread"].max()
    print(f"\n[{label}] Max objective spread across formulations: {max_spread}")

    bad = grouped[grouped["objective_spread"] > 1e-4]
    if not bad.empty:
        print(f"[{label}] WARNING: Some instances have different objectives:")
        print(bad.to_string(index=False))


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if RANDOM_RESULTS_PATH.exists():
        random_df = pd.read_csv(RANDOM_RESULTS_PATH)
        check_objective_consistency(random_df, "random")

        random_summary = summarize_random(random_df)
        summary_path = FIGURE_DIR / "random_summary_table.csv"
        random_summary.to_csv(summary_path, index=False)

        print("\nRandom summary:")
        print(random_summary.to_string(index=False))
        print(f"\nSaved random summary to {summary_path}")

        plot_grouped_bar(
            random_summary,
            value_col="mean_lp_gap",
            ylabel="Mean LP relaxation gap (%)",
            title="LP Relaxation Gap by Formulation on Random Instances",
            output_path=FIGURE_DIR / "random_lp_gap.png",
            percent=True,
        )

        plot_grouped_bar(
            random_summary,
            value_col="mean_runtime",
            ylabel="Mean runtime (seconds, log scale)",
            title="Runtime by Formulation on Random Instances",
            output_path=FIGURE_DIR / "random_runtime.png",
            log_scale=True,
        )

        plot_grouped_bar(
            random_summary,
            value_col="mean_nodes",
            ylabel="Mean branch-and-bound nodes (log scale)",
            title="Branch-and-Bound Nodes by Formulation on Random Instances",
            output_path=FIGURE_DIR / "random_nodes.png",
            log_scale=True,
        )

        plot_grouped_bar(
            random_summary,
            value_col="mean_cuts",
            ylabel="Mean cuts applied",
            title="Cuts Applied by Formulation on Random Instances",
            output_path=FIGURE_DIR / "random_cuts.png",
        )
    else:
        print(f"Random results file not found: {RANDOM_RESULTS_PATH}")

    if ORLIB_RESULTS_PATH.exists():
        orlib_df = pd.read_csv(ORLIB_RESULTS_PATH)
        check_objective_consistency(orlib_df, "orlib")

        orlib_summary = summarize_orlib(orlib_df)
        orlib_summary_path = FIGURE_DIR / "orlib_summary_table.csv"
        orlib_summary.to_csv(orlib_summary_path, index=False)

        orlib_feasible_summary = summarize_orlib_feasible(orlib_df)
        orlib_feasible_summary_path = FIGURE_DIR / "orlib_feasible_summary_table.csv"
        orlib_feasible_summary.to_csv(orlib_feasible_summary_path, index=False)

        print("\nOR-Library summary:")
        print(orlib_summary.to_string(index=False))
        print(f"\nSaved OR-Library summary to {orlib_summary_path}")

        print("\nOR-Library feasible-only summary:")
        print(orlib_feasible_summary.to_string(index=False))
        print(f"\nSaved feasible OR-Library summary to {orlib_feasible_summary_path}")

        infeasible_instances = sorted(
            orlib_df.loc[orlib_df["status"] == "INFEASIBLE", "instance_name"].unique()
        )
        if infeasible_instances:
            print("\nInfeasible OR-Library instances under this formulation:")
            print(", ".join(infeasible_instances))

        plot_orlib_metric(
            orlib_df,
            value_col="runtime_sec",
            ylabel="Runtime (seconds, log scale)",
            title="Runtime by Formulation on Feasible OR-Library Instances",
            output_path=FIGURE_DIR / "orlib_runtime.png",
            log_scale=True,
            only_optimal=True,
        )

        plot_orlib_metric(
            orlib_df,
            value_col="mip_nodes",
            ylabel="Branch-and-bound nodes",
            title="Branch-and-Bound Nodes by Formulation on Feasible OR-Library Instances",
            output_path=FIGURE_DIR / "orlib_nodes.png",
            only_optimal=True,
        )

        plot_orlib_metric(
            orlib_df,
            value_col="cuts_applied",
            ylabel="Cuts applied",
            title="Cuts Applied by Formulation on Feasible OR-Library Instances",
            output_path=FIGURE_DIR / "orlib_cuts.png",
            only_optimal=True,
        )

        plot_orlib_metric(
            orlib_df,
            value_col="lp_gap_fraction",
            ylabel="LP relaxation gap (%)",
            title="LP Relaxation Gap by Formulation on Feasible OR-Library Instances",
            output_path=FIGURE_DIR / "orlib_lp_gap.png",
            percent=True,
            only_optimal=True,
        )
    else:
        print(f"OR-Library results file not found yet: {ORLIB_RESULTS_PATH}")

    print("\nDone. Figures and summary tables saved under figures/.")


if __name__ == "__main__":
    main()