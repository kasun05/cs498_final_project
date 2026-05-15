from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional

import gurobipy as gp
from gurobipy import GRB

from .data_utils import FacilityLocationInstance


@dataclass
class SolveResult:
    instance_name: str
    formulation: str

    num_facilities: int
    num_customers: int
    total_demand: float
    total_capacity: float
    capacity_slack: float

    num_vars: Optional[int]
    num_binary_vars: Optional[int]
    num_constraints: Optional[int]

    objective_value: Optional[float]
    root_lp_objective: Optional[float]
    lp_gap_fraction: Optional[float]
    best_bound: Optional[float]
    mip_gap: Optional[float]
    runtime_sec: Optional[float]
    mip_nodes: Optional[float]
    cuts_applied: Optional[int]
    status: str

    time_limit_sec: Optional[float]
    gurobi_version: str

    def to_dict(self) -> dict:
        return asdict(self)


class CutCounter:
    def __init__(self) -> None:
        self.max_cut_count = 0

    def callback(self, model: gp.Model, where: int) -> None:
        if where == GRB.Callback.MIP:
            current = int(model.cbGet(GRB.Callback.MIP_CUTCNT))
            if current > self.max_cut_count:
                self.max_cut_count = current


def _solve_root_relaxation(model_builder: Callable[[FacilityLocationInstance], gp.Model], instance: FacilityLocationInstance) -> Optional[float]:
    root_model = model_builder(instance)
    root_model = root_model.relax()
    root_model.Params.OutputFlag = 0
    root_model.optimize()

    if root_model.Status == GRB.OPTIMAL:
        return float(root_model.ObjVal)
    return None


def solve_and_measure(
    instance: FacilityLocationInstance,
    formulation_name: str,
    model_builder: Callable[[FacilityLocationInstance], gp.Model],
    time_limit_sec: Optional[float] = None,
) -> SolveResult:
    root_lp_obj = _solve_root_relaxation(model_builder, instance)

    model = model_builder(instance)
    model.Params.OutputFlag = 0
    if time_limit_sec is not None:
        model.Params.TimeLimit = time_limit_sec

    cut_counter = CutCounter()
    model.optimize(cut_counter.callback)

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    status_str = status_map.get(model.Status, str(model.Status))

    objective_value = float(model.ObjVal) if model.SolCount > 0 else None
    runtime_sec = float(model.Runtime) if hasattr(model, "Runtime") else None
    mip_nodes = float(model.NodeCount) if hasattr(model, "NodeCount") else None

    lp_gap_fraction = None
    if objective_value is not None and root_lp_obj is not None and abs(objective_value) > 1e-9:
        lp_gap_fraction = float((objective_value - root_lp_obj) / abs(objective_value))
    
    best_bound = float(model.ObjBound) if model.Status not in [GRB.INFEASIBLE] else None
    mip_gap = float(model.MIPGap) if model.SolCount > 0 else None

    num_vars = int(model.NumVars)
    num_binary_vars = sum(1 for v in model.getVars() if v.VType == GRB.BINARY)
    num_constraints = int(model.NumConstrs + model.NumGenConstrs)

    total_demand = float(instance.demands.sum())
    total_capacity = float(instance.capacities.sum())
    capacity_slack = total_capacity / total_demand if total_demand > 1e-9 else None

    gurobi_version = ".".join(map(str, gp.gurobi.version()))

    return SolveResult(
        instance_name=instance.name,
        formulation=formulation_name,

        num_facilities=instance.num_facilities,
        num_customers=instance.num_customers,
        total_demand=total_demand,
        total_capacity=total_capacity,
        capacity_slack=capacity_slack,

        num_vars=num_vars,
        num_binary_vars=num_binary_vars,
        num_constraints=num_constraints,

        objective_value=objective_value,
        root_lp_objective=root_lp_obj,
        lp_gap_fraction=lp_gap_fraction,
        best_bound=best_bound,
        mip_gap=mip_gap,
        runtime_sec=runtime_sec,
        mip_nodes=mip_nodes,
        cuts_applied=cut_counter.max_cut_count,
        status=status_str,

        time_limit_sec=time_limit_sec,
        gurobi_version=gurobi_version,
    )   
