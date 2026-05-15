from __future__ import annotations

from typing import Dict, Tuple

import gurobipy as gp
from gurobipy import GRB

from .data_utils import FacilityLocationInstance


def _base_model(instance: FacilityLocationInstance, model_name: str) -> tuple[gp.Model, Dict[Tuple[int, int], gp.Var], Dict[int, gp.Var]]:
    model = gp.Model(model_name)
    model.Params.OutputFlag = 0

    I = range(instance.num_customers)
    J = range(instance.num_facilities)

    x = {(i, j): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for i in I for j in J}
    y = {j: model.addVar(vtype=GRB.BINARY, name=f"y_{j}") for j in J}

    model.setObjective(
        gp.quicksum(instance.opening_costs[j] * y[j] for j in J)
        + gp.quicksum(instance.assignment_costs[i, j] * x[i, j] for i in I for j in J),
        GRB.MINIMIZE,
    )

    # Every customer assigned to exactly one facility.
    for i in I:
        model.addConstr(gp.quicksum(x[i, j] for j in J) == 1, name=f"assign_{i}")

    return model, x, y


def build_loose_big_m_model(instance: FacilityLocationInstance) -> gp.Model:
    model, x, y = _base_model(instance, f"{instance.name}_loose_big_m")
    I = range(instance.num_customers)
    J = range(instance.num_facilities)

    total_demand = float(instance.demands.sum())

    for j in J:
        shipped = gp.quicksum(instance.demands[i] * x[i, j] for i in I)
        model.addConstr(shipped <= total_demand * y[j], name=f"activate_loose_{j}")
        model.addConstr(shipped <= instance.capacities[j], name=f"capacity_{j}")

    model.update()
    return model


def build_tight_big_m_model(instance: FacilityLocationInstance) -> gp.Model:
    model, x, y = _base_model(instance, f"{instance.name}_tight_big_m")
    I = range(instance.num_customers)
    J = range(instance.num_facilities)

    for j in J:
        shipped = gp.quicksum(instance.demands[i] * x[i, j] for i in I)
        model.addConstr(shipped <= instance.capacities[j] * y[j], name=f"activate_tight_{j}")

    model.update()
    return model


def build_indicator_model(instance: FacilityLocationInstance) -> gp.Model:
    model, x, y = _base_model(instance, f"{instance.name}_indicator")
    I = range(instance.num_customers)
    J = range(instance.num_facilities)

    for j in J:
        shipped = gp.quicksum(instance.demands[i] * x[i, j] for i in I)
        model.addConstr(shipped <= instance.capacities[j], name=f"capacity_{j}")
        model.addGenConstrIndicator(y[j], 0, shipped == 0.0, name=f"off_zero_flow_{j}")

    model.update()
    return model


FORMULATION_BUILDERS = {
    "loose_big_m": build_loose_big_m_model,
    "tight_big_m": build_tight_big_m_model,
    "indicator": build_indicator_model,
}
