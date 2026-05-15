# Facility Location Formulation Comparison

This project compares three mixed-integer programming formulations for the capacitated facility location problem using Gurobi:

- loose Big-M formulation
- tight Big-M formulation
- indicator-constraint formulation

The goal is to study how different formulations of the same optimization problem affect solver behavior, including LP relaxation gap, runtime, branch-and-bound nodes, and cuts generated.

## Project layout

```text
src/
  data_utils.py        # Random instance generator and OR-Library parser
  models.py            # Gurobi model builders for the three formulations
  metrics.py           # Solve metrics and cut-count callback
  run_experiments.py   # CLI for running one random or OR-Library instance

run_random_grid.py       # Runs the random instance experiment grid
download_orlib.py        # Downloads selected OR-Library instances
run_orlib_experiments.py # Runs OR-Library experiments
plot_results.py          # Generates summary CSVs and plots

results/                 # Experiment CSV outputs
figures/                 # Generated plots and summary tables
benchmarks/              # OR-Library benchmark files