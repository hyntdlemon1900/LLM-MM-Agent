"""
Pyomo Optimization Model Template
==================================

通用Pyomo优化建模模板，用于指导用户构建MILP/LP求解器。

使用方法：
1. 继承 PyomoTemplateSolver 类（参考 KnapsackSolver 示例）
2. 重写 build_model() 定义优化问题（集合、变量、目标、约束）
3. 重写 get_solution() 提取求解结果
4. 调用 build_model() -> solve() -> get_solution()

支持的求解器：gurobi, cplex, cbc, glpk, ipopt
参考资料：https://pyomo.readthedocs.io/
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
from ...problem_template.runtime.BaseTemplateSolver import BaseTemplateSolver
from abc import ABC, abstractmethod
import inspect

class TemplateSolver(BaseTemplateSolver):
    """
    Pyomo优化求解器模板基类

    工作流程：
    1. __init__: 初始化求解器配置
    2. solve(): 求解模型并自动提取解
    """
    
    def __init__(
        self,
        problem_data: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化求解器
        """
        super().__init__(problem_data)
        self.model = None
               
    def _solve_with_pyomo_core(
        self,
        solver_name: str = 'gurobi',
        time_limit: int = 300,
        mip_gap: float = 0.01,
        verbose: bool = False,
    ) -> Optional[Dict[str, Any]]:

        """
        Args:
            problem_data: 问题数据字典（可选，供子类使用）
            solver_name: Pyomo求解器名称 ('gurobi', 'cplex', 'cbc', 'glpk', 'ipopt')
            time_limit: 求解时间限制（秒）
            mip_gap: MIP最优性间隙容差
            verbose: 是否输出求解日志
        """        
        self.solver_name = solver_name
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.verbose = verbose
        
        solver = SolverFactory(self.solver_name)
        if not solver.available():
            raise RuntimeError(f"求解器 '{self.solver_name}' 不可用")
        
        self._configure_solver_options(solver)
         
        try:
            self.results = solver.solve(self.model,tee=self.verbose)
        except Exception as e:
            return 0
      
        success = self._check_solution_status()

        return success
        
    def _configure_solver_options(self, solver):
        """配置求解器参数"""
        if self.solver_name == 'gurobi':
            solver.options['TimeLimit'] = self.time_limit
            solver.options['MIPGap'] = self.mip_gap
            solver.options['LogToConsole'] = 1 if self.verbose else 0
        elif self.solver_name == 'cplex':
            solver.options['timelimit'] = self.time_limit
            solver.options['mip_tolerances_mipgap'] = self.mip_gap
        elif self.solver_name == 'cbc':
            solver.options['seconds'] = self.time_limit
            solver.options['ratioGap'] = self.mip_gap
        elif self.solver_name == 'glpk':
            solver.options['tmlim'] = self.time_limit
            solver.options['mipgap'] = self.mip_gap
    
    def _check_solution_status(self) -> bool:
        """检查求解状态"""
        if self.results is None:
            return 0
        
        term_cond = self.results.solver.termination_condition
        
        if term_cond == TerminationCondition.optimal:
            return 1
        
        if term_cond in [TerminationCondition.feasible, TerminationCondition.maxTimeLimit,
                        TerminationCondition.maxIterations, TerminationCondition.maxEvaluations]:
            return 2
        
        return 0
    
    def _call_with_filtered_kwargs(self, fn, **kwargs):
        """
        只把 fn 能接收的关键字参数传进去，多余的自动丢弃。
        """
        sig = inspect.signature(fn)
        params = sig.parameters

        # 如果目标函数本身就有 **kwargs，那直接全传
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return fn(**kwargs)

        filtered = {}
        for name, param in params.items():
            if name == 'self':
                continue
            if name in kwargs:
                filtered[name] = kwargs[name]
        return fn(**filtered)
    
    def solve(
        self,
        solver_name: str = 'gurobi',
        time_limit: int = 300,
        mip_gap: float = 0.01,
        verbose: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        统一入口：根据是否存在 solve_heuristic 自动选择，
        同时自动裁剪参数，只传递目标函数支持的部分。
        """
        solver_kwargs = dict(
            solver_name=solver_name,
            time_limit=time_limit,
            mip_gap=mip_gap,
            verbose=verbose,
        )

        if hasattr(self, 'solve_with_heuristic') and callable(getattr(self, 'solve_with_heuristic')):
            return self._call_with_filtered_kwargs(self.solve_with_heuristic, **solver_kwargs)
        else:
            return self._call_with_filtered_kwargs(self.solve_with_pyomo, **solver_kwargs)
        
#用户实现------------------------------------------------------------------------------------------------------
    def check_feasibility(self) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validates the feasibility of the current solution stored in self.solution.
        
        Returns:
            Tuple[bool, Dict[str, List[str]]]: 
            - (True, {}): If the solution structure is valid.
            - (False, error_dict): If any constraint is violated, returning the raw dictionary of errors grouped by category.
        """
        if self.solution is None:
            return False, {"General": ["Solution object is None."]}

        errors_by_category = {
            "Budget Violated": [],
            "Invalid Placement": [],
            "Incomplete Routing": [],
            "Invalid Routing Target": [],
            "Data Format Error": []
        }

        # ---------------------------------------------------------------------
        # 1. Check INA Deployment Budget
        # ---------------------------------------------------------------------
        ina_switches = self.solution.get("ina_placement_switches", [])
        if ina_switches is None:
            errors_by_category["Data Format Error"].append("Critical: self.solution['ina_placement_switches'] is None. It MUST be a List[int].")
            distinct_ina_switches = set()
        else:
            distinct_ina_switches = set(ina_switches) # Deduplicate just in case
        
        if len(distinct_ina_switches) > self.ina_budget:
            errors_by_category["Budget Violated"].append(
                f"INA Placement Violation: You selected {len(distinct_ina_switches)} switches, but the budget allows at most {self.ina_budget}."
            )

        # (Optional) Validate that deployed switches are physically capable candidates
        valid_candidates_set = set(self.ina_candidates)
        invalid_deployments = [s for s in distinct_ina_switches if s not in valid_candidates_set]
        if invalid_deployments:
            errors_by_category["Invalid Placement"].append(
                f"Invalid INA Candidate: Switches {invalid_deployments} are not in the valid candidate set."
            )

        # ---------------------------------------------------------------------
        # 2. Check Routing Logic (Worker Assignments)
        # ---------------------------------------------------------------------
        # Standard: List[List[int]].
        worker_agg_id = self.solution.get("worker_agg_id", [])
        if worker_agg_id is None:
            errors_by_category["Data Format Error"].append("Critical: self.solution['worker_agg_id'] is None. It MUST be a List[List[int]].")
            return False, errors_by_category

        if not isinstance(worker_agg_id, list):
             errors_by_category["Data Format Error"].append("Critical: self.solution['worker_agg_id'] MUST be a List[List[int]], but got something else.")
             return False, errors_by_category

        for j in range(self.jobs_num):
            # A. Check Job Existence in Solution
            if j >= len(worker_agg_id):
                errors_by_category["Incomplete Routing"].append(f"Job {j} is missing from 'worker_agg_id' (List index out of range).")
                continue
            
            job_assignments = worker_agg_id[j]
            num_workers = self.workers_num[j]
            job_ps = self.ps_id[j]
            
            # B. Check Worker Coverage
            if len(job_assignments) != num_workers:
                errors_by_category["Incomplete Routing"].append(
                    f"Job ID {j} Assignment Count Mismatch: Expected assignments for {num_workers} workers, but found {len(job_assignments)} assignments."
                )
            
            for w_idx in range(num_workers):
                # Retrieve assigned aggregation node safely
                try:
                    agg_node = job_assignments[w_idx]
                except (IndexError, TypeError):
                     errors_by_category["Incomplete Routing"].append(f"Job ID {j}, Worker ID {w_idx} assignment is missing.")
                     continue
                
                # -----------------------------------------------------------------
                # 3. Check Aggregation Point Validity
                # -----------------------------------------------------------------
                is_direct_to_ps = (agg_node == job_ps)
                is_via_ina_switch = (agg_node in distinct_ina_switches)
                
                if not (is_direct_to_ps or is_via_ina_switch):
                    errors_by_category["Invalid Routing Target"].append(
                        f"Job ID {j} Worker ID {w_idx} assigned to Node ID {agg_node}, which is neither the Job PS (Node ID {job_ps}) nor any selected INA switch."
                    )
        
        # Filter empty categories
        final_errors = {k: v for k, v in errors_by_category.items() if v}

        if not final_errors:
            return True, {}
        else:
            return False, final_errors

    def _get_flows_on_link(
        self,
        edge: Tuple[int, int],
        src_set: List[int],
        dst_set: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Identifies which flows (src_idx -> dst_idx) traverse the given edge.
        Used to construct Link Bandwidth Constraints.
        
        Returns:
            List of (src_index, dst_index) tuples.
        """
        flows = []
        edge_rev = (edge[1], edge[0])
        
        for s_idx, s_node in enumerate(src_set):
            for d_idx, d_node in enumerate(dst_set):
                path_edges = self._get_path_links(s_node, d_node)
                # Check if the edge (or its reverse) is part of the flow's path
                if edge in path_edges or edge_rev in path_edges:
                    flows.append((s_idx, d_idx))
        return flows
    
    def build_model(self) -> None:
        """
        Build the full Pyomo MILP model for the LLMINA problem.

        Mathematical entities:

            Sets:
                J      : job indices
                S      : INA candidate indices
                W_align: aligned worker index set (0..max workers per job - 1)
                W[j]   : actual worker indices of job j (0..workers_num[j] - 1)

            Variables:
                x_s        : binary INA placement indicator on candidate s
                y_jws      : binary worker→INA assignment
                              (1 if worker w of job j uses INA s)
                y_jwd      : binary worker→PS assignment
                              (1 if worker w of job j sends directly to PS)
                gamma_j    : effective rate of job j
                gamma_jws  : rate from worker w to INA s for job j
                gamma_jwd  : rate from worker w to PS for job j
                gamma_js   : rate from INA s to PS for job j
                alpha      : 1 / makespan (objective variable)

            Objective:
                Maximize alpha (equivalently minimize makespan).

            Constraints (C1–C12):
                C1  INA budget
                C2  each real worker chooses exactly one aggregation point
                C3  worker uses INA only if it is deployed
                C4  dummy workers (padded indices) have zero rates
                C5  big-M coupling between assignment and rates
                C6  INA processing capacity
                C7  worker rate consistency with job rate
                C8  INA egress not smaller than any inbound worker rate
                C9  INA flow conservation
                C10 PS inbound rate covers job rate
                C11 makespan definition: alpha <= gamma_j / job_size_j
                C12 per-link capacity constraints
        """
        jobs_num = self.problem_data["jobs_num"]
        workers_num = self.problem_data["instance"]["workers_num"]
        workers_num_align = max(workers_num)
        jobs_size = self.problem_data["instance"]["jobs_size"]
        workers_id = self.problem_data["instance"]["workers_id"]
        ps_id = self.problem_data["instance"]["ps_id"]
        ina_budget= self.problem_data["ina_budget"]
        Cs = self.problem_data["Cs"]
        Ps = self.problem_data["network"].basic_band

        self.sd_id: List[int] = []
        for j in range(self.jobs_num):
            ps_node = self.ps_id[j]
            neighbors = list(self.G.neighbors(ps_node))
            if not neighbors:
                raise ValueError(f"PS node {ps_node} is isolated!")
            self.sd_id.append(neighbors[0]) # The unique ToR switch for this PS
        model = pyo.ConcreteModel(name="LLMINA_INA_Placement_and_Routing")

        # ----- Sets -----
        model.J = pyo.RangeSet(0, jobs_num - 1)
        model.S = pyo.RangeSet(0, self.ina_candidates_num - 1)
        model.W_align = pyo.RangeSet(0, workers_num_align - 1)

        # Job-specific worker sets (only real workers; dummy indices are >= workers_num[j]).
        model.W = {j: pyo.RangeSet(0, workers_num[j] - 1) for j in range(jobs_num)}

        # ----- Decision variables -----

        # INA placement on candidate switches.
        model.x_s = pyo.Var(model.S, domain=pyo.Binary, doc="INA placement on switch s")

        # Worker→INA assignment.
        model.y_jws = pyo.Var(
            model.J, model.W_align, model.S,
            domain=pyo.Binary,
            doc="Worker w of job j uses INA s",
        )

        # Worker→PS direct assignment.
        model.y_jwd = pyo.Var(
            model.J, model.W_align,
            domain=pyo.Binary,
            doc="Worker w of job j directly uses PS",
        )

        # Effective job rates.
        model.gamma_j = pyo.Var(
            model.J,
            domain=pyo.NonNegativeReals,
            doc="Effective rate of job j",
        )

        # Worker→INA rates.
        model.gamma_jws = pyo.Var(
            model.J, model.W_align, model.S,
            domain=pyo.NonNegativeReals,
            doc="Rate worker w → INA s for job j",
        )

        # Worker→PS rates.
        model.gamma_jwd = pyo.Var(
            model.J, model.W_align,
            domain=pyo.NonNegativeReals,
            doc="Rate worker w → PS for job j",
        )

        # INA→PS rates.
        model.gamma_js = pyo.Var(
            model.J, model.S,
            domain=pyo.NonNegativeReals,
            doc="Rate INA s → PS for job j",
        )

        # Inverse makespan.
        model.alpha = pyo.Var(
            domain=pyo.NonNegativeReals,
            doc="Inverse of makespan (1 / makespan)",
        )

        # ----- Objective -----

        def obj_rule(m):
            # Maximize alpha, which is 1 / makespan.
            return m.alpha

        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

        # ----- Constraints -----

        # C1: INA budget.
        def ina_budget_rule(m):
            return sum(m.x_s[s] for s in m.S) <= ina_budget

        model.ina_budget = pyo.Constraint(rule=ina_budget_rule)

        # C2: each real worker chooses exactly one aggregation point (INA or PS).
        def worker_choice_rule(m, j, w):
            if w >= workers_num[j]:
                # Padded dummy worker index: skip constraint.
                return pyo.Constraint.Skip
            return sum(m.y_jws[j, w, s] for s in m.S) + m.y_jwd[j, w] == 1

        model.worker_choice = pyo.Constraint(model.J, model.W_align, rule=worker_choice_rule)

        # C3: worker can use INA s only if s is deployed.
        def ina_deployment_rule(m, j, w, s):
            return m.y_jws[j, w, s] <= m.x_s[s]

        model.ina_deployment = pyo.Constraint(model.J, model.W_align, model.S, rule=ina_deployment_rule)

        # C4: dummy workers (padded indices) have zero rates to INA.
        def zero_dummy_workers_ina_rule(m, j, w, s):
            if w >= workers_num[j]:
                return m.gamma_jws[j, w, s] == 0
            return pyo.Constraint.Skip

        model.zero_dummy_ina = pyo.Constraint(
            model.J, model.W_align, model.S,
            rule=zero_dummy_workers_ina_rule,
        )

        # C4 (PS side): dummy workers have zero rates to PS.
        def zero_dummy_workers_ps_rule(m, j, w):
            if w >= workers_num[j]:
                return m.gamma_jwd[j, w] == 0
            return pyo.Constraint.Skip

        model.zero_dummy_ps = pyo.Constraint(
            model.J, model.W_align,
            rule=zero_dummy_workers_ps_rule,
        )

        # C5: big-M coupling between worker→INA assignment and rate.
        M_ina = Cs  # Upper bound for worker→INA rate.
        def bigm_ina_rule(m, j, w, s):
            return m.gamma_jws[j, w, s] <= M_ina * m.y_jws[j, w, s]

        model.bigm_ina = pyo.Constraint(model.J, model.W_align, model.S, rule=bigm_ina_rule)

        # C5: big-M coupling between worker→PS assignment and rate.
        M_ps = Ps  # Upper bound for worker→PS rate.
        def bigm_ps_rule(m, j, w):
            return m.gamma_jwd[j, w] <= M_ps * m.y_jwd[j, w]

        model.bigm_ps = pyo.Constraint(model.J, model.W_align, rule=bigm_ps_rule)

        # C6: INA processing capacity.
        def ina_capacity_rule(m, s):
            return (
                sum(m.gamma_jws[j, w, s] for j in m.J for w in range(workers_num[j]))
                <= Cs * m.x_s[s]
            )

        model.ina_capacity = pyo.Constraint(model.S, rule=ina_capacity_rule)

        # C7: per-worker rate consistency with job rate.
        def worker_rate_rule(m, j, w):
            if w >= workers_num[j]:
                return pyo.Constraint.Skip
            return (
                sum(m.gamma_jws[j, w, s] for s in m.S) + m.gamma_jwd[j, w] >= m.gamma_j[j]
            )

        model.worker_rate = pyo.Constraint(model.J, model.W_align, rule=worker_rate_rule)

        # C8: INA egress rate not smaller than any inbound worker rate.
        def egress_consistency_rule(m, j, w, s):
            if w >= workers_num[j]:
                return pyo.Constraint.Skip
            return m.gamma_js[j, s] >= m.gamma_jws[j, w, s]

        model.egress_consistency = pyo.Constraint(
            model.J, model.W_align, model.S,
            rule=egress_consistency_rule,
        )

        # C9: INA flow conservation (egress not larger than inbound sum).
        def ina_flow_conservation_rule(m, j, s):
            return m.gamma_js[j, s] <= sum(
                m.gamma_jws[j, w, s] for w in range(workers_num[j])
            )

        model.ina_flow_conservation = pyo.Constraint(
            model.J, model.S, rule=ina_flow_conservation_rule
        )

        # C10: PS inbound rate must cover job rate.
        def ps_inflow_rule(m, j):
            return m.gamma_j[j] <= (
                sum(m.gamma_js[j, s] for s in m.S)
                + sum(m.gamma_jwd[j, w] for w in range(workers_num[j]))
            )

        model.ps_inflow = pyo.Constraint(model.J, rule=ps_inflow_rule)

        # C11: makespan definition via per-job rate:
        #      alpha <= gamma_j / job_size_j for each job j.
        def makespan_rule(m, j):
            return m.alpha <= m.gamma_j[j] / jobs_size[j]

        model.makespan = pyo.Constraint(model.J, rule=makespan_rule)

        # C12: link capacity constraints.
        # Precompute, for each logical link, which flows (per job) traverse it.
        link_flows: Dict[Any, Any] = {}
        for edge, capacity in self.bandwidth_mapping.items():
            flows_per_job = {}
            for j in range(jobs_num):
                # For job j, workers_id[j] is a flat list of worker node IDs.
                w_to_s = self._get_flows_on_link(edge, workers_id[j], self.ina_candidates)
                w_to_ps = self._get_flows_on_link(edge, workers_id[j], [ps_id[j]])
                s_to_ps = self._get_flows_on_link(edge, self.ina_candidates, [ps_id[j]])
                flows_per_job[j] = (w_to_s, w_to_ps, s_to_ps)

            link_flows[edge] = (capacity, flows_per_job)

            def link_capacity_rule(m, u, v):
                """
                Enforce capacity on link (u, v).

                The rule is indexed by (u, v), which reconstructs the edge tuple.
                For PS last-hop edges, the effective capacity is Ps instead of
                the base link capacity.
                """
                edge = (u, v)
                base_capacity, flows_per_job = link_flows[edge]

                # If this edge is the last hop to/from some job's PS, use Ps.
                cap = base_capacity
                for j in range(jobs_num):
                    if edge == (self.sd_id[j], ps_id[j]) or edge == (ps_id[j], self.sd_id[j]):
                        cap = Ps
                        break

                # Initialize a Pyomo expression for total load using 0 * m.alpha.
                total_load = 0 * m.alpha

                for j in range(jobs_num):
                    w_to_s, w_to_ps, s_to_ps = flows_per_job[j]

                    # Worker→INA flows traversing this edge.
                    for w_idx, s_idx in w_to_s:
                        total_load += m.gamma_jws[j, w_idx, s_idx]

                    # Worker→PS flows traversing this edge.
                    for w_idx, _ in w_to_ps:
                        total_load += m.gamma_jwd[j, w_idx]

                    # INA→PS flows traversing this edge.
                    for s_idx, _ in s_to_ps:
                        total_load += m.gamma_js[j, s_idx]

                return total_load <= cap

        model.link_capacity = pyo.Constraint(
            self.bandwidth_mapping.keys(), rule=link_capacity_rule
        )

        self.model = model

    # ========= MILP solve & solution extraction =========
    def solve_with_pyomo(
        self,
        solver_name: str = "gurobi",
        time_limit: int = 300,
        mip_gap: float = 0.01,
        verbose: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Solve the full MILP using the parent class core routine, then extract:
            - INA placement (switch IDs),
            - worker aggregation decision for each job and worker.

        Returns:
            solution:
                Dict with keys:
                - "ina_placement_switches": List[int], deployed INA switch IDs.
                - "worker_agg_id": Dict[job][worker] -> aggregation point ID (INA or PS).
                None if the MILP is infeasible or not solved.
        """
        solved = super()._solve_with_pyomo_core(
            solver_name=solver_name,
            time_limit=time_limit,
            mip_gap=mip_gap,
            verbose=verbose,
        )

        if not solved:
            return None

        m = self.model
        jobs_num = self.problem_data["jobs_num"]

        # INA placement (decode x_s into physical switch IDs).
        ina_binary = [int(round(pyo.value(m.x_s[s]))) for s in range(self.ina_candidates_num)]
        ina_placement_switches = [
            self.ina_candidates[s]
            for s in range(self.ina_candidates_num)
            if ina_binary[s] > 0.5
        ]

        # Worker aggregation point: either PS or some INA.
        workers_num = self.problem_data["instance"]["workers_num"]
        ps_id = self.problem_data["instance"]["ps_id"]

        # worker_agg_id[j][w] is the node ID of the aggregation point for worker w of job j.
        # Structure: List[List[int]] (Consistent with OUTPUT_FUNCTION_TEMPLATE)
        worker_agg_id: List[List[int]] = []

        for j in range(jobs_num):
            job_assignments = []
            for w in range(workers_num[j]):
                if pyo.value(m.y_jwd[j, w]) > 0.5:
                    # Worker sends directly to PS.
                    job_assignments.append(ps_id[j])
                else:
                    # Find the INA s with y_jws[j, w, s] = 1.
                    found_s = -1
                    for s in range(self.ina_candidates_num):
                        if pyo.value(m.y_jws[j, w, s]) > 0.5:
                            found_s = self.ina_candidates[s]
                            break
                    
                    if found_s == -1:
                        # Should not happen in a valid solution, but as a fallback
                        found_s = ps_id[j] 
                    
                    job_assignments.append(found_s)
            worker_agg_id.append(job_assignments)

        self.worker_agg_id = worker_agg_id
        self.solution = {
            "ina_placement_switches": ina_placement_switches,
            "worker_agg_id": worker_agg_id,
        }
        return self.solution

    def get_makespan(self, ina_placement_switches, worker_agg_id) -> Tuple[float, Dict[int, float]]:
        """
        Compute makespan for a given discrete decision (INA placement and worker assignments).

        Steps:
            1) Unfix any previous binary decisions.
            2) Fix x_s and y_jws/y_jwd according to the given deployment and assignments.
            3) Resolve the resulting LP to optimize continuous rate variables and compute makespan.
            4) Return makespan = 1 / alpha.

        Args:
            ina_placement_switches (List[int]): List of deployed INA switch IDs.
            worker_agg_id (List[List[int]]): 
                Assignments mapping Job -> Worker -> Aggregation Node ID.
        
        Raises:
            ValueError: If the configuration is infeasible (structure check).
            RuntimeError: If the LP solver fails or returns an invalid makespan.

        This function assumes self.build_model() has been called at least once.
        """
        if self.model is None:
            self.build_model()

        # Update self.solution to check feasibility
        self.solution = {
            "ina_placement_switches": ina_placement_switches,
            "worker_agg_id": worker_agg_id,
        }
        
        is_feasible, errors = self.check_feasibility()
        
        if not is_feasible:
             final_msg_lines = ["Feasibility Check Failed. Failed to retrieve makespan."]
             for cat, msgs in errors.items():
                 final_msg_lines.append(f"[{cat}]: {len(msgs)} issues.")
                 # Show top 3 unique errors
                 for m in msgs[:3]:
                     final_msg_lines.append(f"    - {m}")
                 if len(msgs) > 3:
                     final_msg_lines.append(f"    - ... and {len(msgs)-3} more similar errors.")
             
             raise ValueError("\n".join(final_msg_lines))

        m = self.model
        jobs_num = self.problem_data["jobs_num"]
        workers_num = self.problem_data["instance"]["workers_num"]
        workers_num_align = max(workers_num)
        ps_id = self.problem_data["instance"]["ps_id"]

        deployed_set = set(ina_placement_switches)

        # Step 0: unfix previous discrete decisions (for repeated evaluations).
        for s in m.S:
            m.x_s[s].unfix()
        for j in range(jobs_num):
            for w in range(workers_num_align):
                m.y_jwd[j, w].unfix()
                for s in m.S:
                    m.y_jws[j, w, s].unfix()

        # Step 1: fix INA placement x_s to match ina_placement_switches.
        for s in m.S:
            sw_id = self.ina_candidates[s]
            m.x_s[s].fix(1 if sw_id in deployed_set else 0)

        # Step 2: fix worker aggregation decisions (including dummy workers).
        for j in range(jobs_num):
            real_w = workers_num[j]

            for w in range(workers_num_align):
                if w >= real_w:
                    # Dummy worker: force all assignments to 0.
                    m.y_jwd[j, w].fix(0)
                    for s in m.S:
                        m.y_jws[j, w, s].fix(0)
                    continue

                agg_point = worker_agg_id[j][w]

                if agg_point == ps_id[j]:
                    # Direct PS assignment.
                    m.y_jwd[j, w].fix(1)
                    for s in m.S:
                        m.y_jws[j, w, s].fix(0)
                else:
                    # Assignment to a specific INA switch.
                    m.y_jwd[j, w].fix(0)
                    for s in m.S:
                        m.y_jws[j, w, s].fix(
                            1 if self.ina_candidates[s] == agg_point else 0
                        )

        # Step 3: solve the LP with fixed discrete decisions.
        # solved = super()._solve_with_pyomo_core(
        solved = self._solve_with_pyomo_core(
            solver_name="gurobi",
            time_limit=300,
            mip_gap=0.0,  # exact LP (no integrality).
            verbose=True,
        )
        if not solved:
            raise RuntimeError("LP Solver failed to find an optimal solution (Solver Status check failed).")

        alpha_value = pyo.value(m.alpha)
        if alpha_value is None or alpha_value <= 1e-9:
            raise RuntimeError(f"Invalid alpha value derived: {alpha_value}. The model might be infeasible or unbounded.")

        # Update model with the latest solution and return the implied makespan.
        self.model = m
        
        gamma_j_values = {j: pyo.value(m.gamma_j[j]) for j in m.J}
        makespan = 1.0 / alpha_value

        return makespan, gamma_j_values

OUTPUT_FUNCTION_TEMPLATE = """
    def solve_with_heuristic(self) -> Optional[Dict[str, Any]]:
        '''
        to be implemented below: solve the MILP model with heuristic methods
        '''         
        # Initialize the solution structure
        self.solution = {
            # 1. INA Placement (The "Where")
            # A simple list of Switch IDs selected for INA.
            "ina_placement_switches": [], 
            
            # 2. Worker Routing (The "How")
            # A Nested List (Jagged Array) mirroring the structure of `self.workers_id`.
            # Rule: self.worker_agg_id[j][w] is the target node for self.workers_id[j][w].
            #
            # Structure: List[List[int]]
            #   - Outer List: Corresponds to each Job (Index j)
            #   - Inner List: Corresponds to each Worker in that Job (Index w) in strict order.
            #   - Value: The integer Node ID of the chosen aggregation switch (or PS).
            "worker_agg_id": [] 
        }
        
        return self.solution
"""