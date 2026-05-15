from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class FacilityLocationInstance:
    name: str
    num_facilities: int
    num_customers: int
    capacities: np.ndarray          # shape (m,)
    opening_costs: np.ndarray       # shape (m,)
    demands: np.ndarray             # shape (n,)
    assignment_costs: np.ndarray    # shape (n, m)

    def validate(self) -> None:
        if self.capacities.shape != (self.num_facilities,):
            raise ValueError("capacities shape mismatch")
        if self.opening_costs.shape != (self.num_facilities,):
            raise ValueError("opening_costs shape mismatch")
        if self.demands.shape != (self.num_customers,):
            raise ValueError("demands shape mismatch")
        if self.assignment_costs.shape != (self.num_customers, self.num_facilities):
            raise ValueError("assignment_costs shape mismatch")
        if np.any(self.capacities <= 0):
            raise ValueError("all capacities must be positive")
        if np.any(self.demands <= 0):
            raise ValueError("all demands must be positive")


class TokenReader:
    def __init__(self, text: str):
        self.tokens: List[str] = text.split()
        self.idx = 0

    def next_int(self) -> int:
        if self.idx >= len(self.tokens):
            raise ValueError("unexpected end of file while parsing OR-Library instance")
        value = int(float(self.tokens[self.idx]))
        self.idx += 1
        return value

    def next_float(self) -> float:
        if self.idx >= len(self.tokens):
            raise ValueError("unexpected end of file while parsing OR-Library instance")
        value = float(self.tokens[self.idx])
        self.idx += 1
        return value


def generate_random_instance(
    num_facilities: int,
    num_customers: int,
    seed: int = 0,
    capacity_slack: float = 1.2,
) -> FacilityLocationInstance:
    """
    Generate a random capacitated facility location instance.

    Facilities and customers are placed in the unit square. Assignment costs are
    demand-weighted Euclidean distances plus a small random perturbation.
    Capacities are scaled so that total capacity exceeds total demand.
    """
    rng = np.random.default_rng(seed)

    facility_xy = rng.uniform(0.0, 1.0, size=(num_facilities, 2))
    customer_xy = rng.uniform(0.0, 1.0, size=(num_customers, 2))

    demands = rng.integers(5, 31, size=num_customers).astype(float)
    total_demand = float(demands.sum())

    raw_capacity_weights = rng.uniform(0.5, 1.5, size=num_facilities)
    capacities = raw_capacity_weights / raw_capacity_weights.sum() * total_demand * capacity_slack
    capacities = np.ceil(capacities)

    opening_costs = rng.integers(80, 301, size=num_facilities).astype(float)

    distances = np.linalg.norm(customer_xy[:, None, :] - facility_xy[None, :, :], axis=2)
    assignment_costs = 10.0 * distances + rng.uniform(0.0, 2.0, size=(num_customers, num_facilities))
    assignment_costs = np.round(assignment_costs, 3)

    instance = FacilityLocationInstance(
        name=f"random_f{num_facilities}_c{num_customers}_s{seed}",
        num_facilities=num_facilities,
        num_customers=num_customers,
        capacities=capacities.astype(float),
        opening_costs=opening_costs.astype(float),
        demands=demands.astype(float),
        assignment_costs=assignment_costs.astype(float),
    )
    instance.validate()
    return instance


def load_orlib_cap_instance(path: str | Path) -> FacilityLocationInstance:
    """
    Parse an OR-Library 'cap' style facility-location instance.

    Expected token order:
    - first two integers: number of facilities (m), number of customers (n)
    - for each facility: capacity, opening_cost
    - for each customer:
        demand
        then n_facilities assignment costs
    """
    path = Path(path)
    reader = TokenReader(path.read_text())

    num_facilities = reader.next_int()
    num_customers = reader.next_int()

    capacities = np.zeros(num_facilities, dtype=float)
    opening_costs = np.zeros(num_facilities, dtype=float)
    for j in range(num_facilities):
        capacities[j] = reader.next_float()
        opening_costs[j] = reader.next_float()

    demands = np.zeros(num_customers, dtype=float)
    assignment_costs = np.zeros((num_customers, num_facilities), dtype=float)
    for i in range(num_customers):
        demands[i] = reader.next_float()
        for j in range(num_facilities):
            assignment_costs[i, j] = reader.next_float()

    instance = FacilityLocationInstance(
        name=path.stem,
        num_facilities=num_facilities,
        num_customers=num_customers,
        capacities=capacities,
        opening_costs=opening_costs,
        demands=demands,
        assignment_costs=assignment_costs,
    )
    instance.validate()
    return instance
