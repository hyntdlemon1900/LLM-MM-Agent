
"""
MILP Constraint Analyzer for LLMINA
====================================

This analyzer inspects the limited solution dictionary returned by
`ModelSolver.solve_with_pyomo`. The solver currently exposes only the
following fields inside `solver.solution`:

- `ina_placement_switches`: list of deployed INA switch IDs
- `worker_to_ina`: assignment tensor [job][worker_align][candidate]
- `worker_to_ps`: assignment matrix [job][worker_align]
- `job_rates`: list of achieved job rates (Gbps)
- `makespan`: overall schedule makespan (seconds)

Given that no continuous rate tensors (`gamma_*`) or Pyomo model objects
are available at this stage, the analyzer focuses on constraint checks
that can be derived purely from those discrete outputs. The remaining
constraint families are marked as "not available" with an explanation so
callers can understand the coverage gap.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .ModelSolver import ModelSolver


class ConstraintAnalyzer:
    """Performs lightweight consistency checks on solver outputs."""

    REQUIRED_SOLUTION_KEYS = [
        'ina_placement_switches',
        'worker_to_ina',
        'worker_to_ps',
        'job_rates',
        'makespan',
    ]

    RATE_DATA_REASON = (
        'Requires continuous rate tensors (gamma*) from the Pyomo model, '
        'which are not present in solver.solution.'
    )

    LINK_DATA_REASON = (
        'Requires per-link flow measurements, which are not available in the '
        'discrete solution output.'
    )

    INA_ASSIGNMENT_REASON = (
        'Requires per-candidate INA assignment tensor; solver only provides aggregate worker-to-INA flags.'
    )

    CONSTRAINT_ORDER = [
        'ina_budget',
        'worker_choice',
        'ina_deployment',
        'zero_dummy',
        'bigm_coupling',
        'ina_capacity',
        'worker_rate',
        'egress_consistency',
        'ina_flow_conservation',
        'ps_inflow',
        'makespan',
        'link_capacity',
    ]

    def __init__(self, solver: ModelSolver, epsilon: float = 1e-4) -> None:
        self.solver = solver
        self.epsilon = epsilon
        self.solution: Dict[str, Any] = solver.solution or {}
        self._validate_solution_fields()

        self.instance = solver.instance
        self.K = solver.K
        self.jobs_num = solver.jobs_num
        self.jobs_size = self.instance.get('jobs_size', [])
        self.workers_num = self.instance.get('workers_num', [])
        self.workers_num_align = max(self.workers_num) if self.workers_num else 0
        self.ina_candidates: List[int] = getattr(solver, 'ina_candidates', [])
        self.ina_candidates_num = len(self.ina_candidates)

        self.worker_to_ina_raw = self.solution['worker_to_ina']
        self.worker_to_ps = self.solution['worker_to_ps']
        self._has_candidate_assignments = self._detect_candidate_assignments()
        self.job_rates: List[float] = self.solution['job_rates']
        self.makespan: float = self.solution['makespan']
        self.deployed_switches = set(self.solution['ina_placement_switches'])
        self.Cs = self._coerce_float(getattr(solver, 'Cs', self.instance.get('Cs', 0.0)))
        self.Ps = self._coerce_float(getattr(solver, 'Ps', self.instance.get('Ps', 0.0)))

        self.worker_ps_assignments: List[List[float]] = []
        self.worker_ina_assignments: List[List[List[int]]] = []
        self.worker_rate_matrix: List[List[float]] = []
        self.job_ps_loads: List[float] = []
        self.job_ina_loads: List[float] = []
        self.job_active_workers: List[int] = []
        self.job_switch_loads: List[Dict[int, float]] = []
        self.switch_loads: List[float] = []
        self._prepare_worker_views()

        self.analysis: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_all_constraints(self) -> Dict[str, Any]:
        """Run every constraint check (or mark it as unavailable)."""
        analysis: Dict[str, Dict[str, Any]] = {}

        analysis['ina_budget'] = self._analyze_ina_budget()
        analysis['worker_choice'] = self._analyze_worker_choice()
        analysis['ina_deployment'] = self._analyze_ina_deployment()
        analysis['zero_dummy'] = self._analyze_zero_dummy()
        analysis['bigm_coupling'] = self._analyze_bigm_coupling()
        analysis['ina_capacity'] = self._analyze_ina_capacity()
        analysis['worker_rate'] = self._analyze_worker_rate()
        analysis['egress_consistency'] = self._analyze_egress_consistency()
        analysis['ina_flow_conservation'] = self._analyze_ina_flow_conservation()
        analysis['ps_inflow'] = self._analyze_ps_inflow()
        analysis['makespan'] = self._analyze_makespan()
        analysis['link_capacity'] = self._analyze_link_capacity()

        self.analysis = analysis
        self._print_summary()
        return analysis

    def export_report(self, filepath: str = 'constraint_analysis_report.txt') -> None:
        """Export a text report with the collected analysis results."""
        lines = [
            '=' * 80,
            'LLMINA MILP CONSTRAINT ANALYSIS REPORT',
            '=' * 80,
            f'Makespan: {self.makespan:.4f}',
            f'INA switches deployed: {len(self.deployed_switches)}/{self.K}',
            f'Epsilon threshold: {self.epsilon}',
            '',
        ]

        for key in self.CONSTRAINT_ORDER:
            section_lines = self._format_report_section(key, self.analysis.get(key, {}))
            lines.extend(section_lines)
            lines.append('')

        Path(filepath).write_text('\n'.join(lines), encoding='utf-8')
        print(f'✓ Detailed report exported to: {filepath}')

    # ------------------------------------------------------------------
    # Individual analyses
    # ------------------------------------------------------------------
    def _analyze_ina_budget(self) -> Dict[str, Any]:
        deployed = len(self.deployed_switches)
        budget = self.K
        slack = budget - deployed
        status = 'tight' if slack < self.epsilon else 'slack'
        return {
            'status': status,
            'deployed': deployed,
            'budget': budget,
            'slack': slack,
            'utilization': deployed / budget if budget > 0 else 0.0,
        }

    def _analyze_worker_choice(self) -> Dict[str, Any]:
        tight = 0
        violations: List[Dict[str, Any]] = []
        total_checked = 0

        for j in range(self.jobs_num):
            worker_count = self.workers_num[j] if j < len(self.workers_num) else 0
            ps_row = self.worker_ps_assignments[j] if j < len(self.worker_ps_assignments) else []
            ina_row = self.worker_ina_assignments[j] if j < len(self.worker_ina_assignments) else []
            for w in range(worker_count):
                ps_flag = ps_row[w] > self.epsilon if w < len(ps_row) else False
                ina_flag = bool(ina_row[w]) if w < len(ina_row) else False
                total = (1.0 if ps_flag else 0.0) + (1.0 if ina_flag else 0.0)

                total_checked += 1
                if abs(total - 1.0) > self.epsilon:
                    violations.append({
                        'job': j,
                        'worker': w,
                        'total': total,
                        'ps_choice': 1.0 if ps_flag else 0.0,
                        'ina_choice': 1.0 if ina_flag else 0.0,
                    })
                else:
                    tight += 1

        status = 'ok' if not violations else 'violations'
        return {
            'status': status,
            'tight_count': tight,
            'total_count': total_checked,
            'violations': violations,
        }

    def _analyze_ina_deployment(self) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        total_assignments = 0

        for j in range(self.jobs_num):
            ina_row = self.worker_ina_assignments[j] if j < len(self.worker_ina_assignments) else []
            for w, candidates in enumerate(ina_row):
                for cand_idx in candidates:
                    total_assignments += 1
                    switch_id = self._candidate_id(cand_idx)
                    if switch_id not in self.deployed_switches:
                        violations.append({
                            'job': j,
                            'worker': w,
                            'candidate_index': cand_idx,
                            'switch_id': switch_id,
                        })

        status = 'ok' if not violations else 'violations'
        return {
            'status': status,
            'violations': violations,
            'total_assignments': total_assignments,
        }

    def _analyze_makespan(self) -> Dict[str, Any]:
        makespan = self.makespan
        job_details: List[Dict[str, Any]] = []
        tight_jobs: List[int] = []
        slack_jobs: List[int] = []

        for j, rate in enumerate(self.job_rates):
            size = self.jobs_size[j] if j < len(self.jobs_size) else 0.0
            job_time = size / rate if rate > 1e-9 else float('inf')
            slack = job_time - makespan
            is_tight = abs(slack) < self.epsilon
            job_details.append({
                'job': j,
                'rate': rate,
                'size': size,
                'time': job_time,
                'slack': slack,
                'is_tight': is_tight,
            })
            if is_tight:
                tight_jobs.append(j)
            else:
                slack_jobs.append(j)

        alpha_estimate = 1.0 / makespan if makespan > 1e-9 else None
        return {
            'status': 'analyzed',
            'makespan': makespan,
            'alpha_estimate': alpha_estimate,
            'tight_jobs': tight_jobs,
            'slack_jobs': slack_jobs,
            'details': job_details,
        }


    def _analyze_zero_dummy(self) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        for j in range(self.jobs_num):
            actual_workers = self.workers_num[j] if j < len(self.workers_num) else 0
            for w in range(actual_workers, self.workers_num_align):
                ps_val = self._worker_ps_raw_value(j, w)
                ina_entry = self._get_worker_ina_entry(j, w)
                if abs(ps_val) > self.epsilon or self._entry_has_assignment(ina_entry):
                    violations.append({
                        'job': j,
                        'worker': w,
                        'ps_value': ps_val,
                        'ina_entry': ina_entry,
                    })
        status = 'ok' if not violations else 'violations'
        return {
            'status': status,
            'violations': violations,
            'dummy_workers_checked': self.jobs_num * max(0, self.workers_num_align - min(self.workers_num or [0])),
        }

    def _analyze_bigm_coupling(self) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        max_ina_rate = 0.0
        max_ps_rate = 0.0

        for j in range(self.jobs_num):
            ps_row = self.worker_ps_assignments[j] if j < len(self.worker_ps_assignments) else []
            ina_row = self.worker_ina_assignments[j] if j < len(self.worker_ina_assignments) else []
            rate_row = self.worker_rate_matrix[j] if j < len(self.worker_rate_matrix) else []
            for w in range(len(rate_row)):
                rate = rate_row[w]
                ps_flag = ps_row[w] > self.epsilon if w < len(ps_row) else False
                ina_flag = bool(ina_row[w]) if w < len(ina_row) else False
                if rate <= self.epsilon:
                    continue
                if ps_flag and ina_flag:
                    violations.append({'job': j, 'worker': w, 'reason': 'duplicate_assignments'})
                if ps_flag:
                    max_ps_rate = max(max_ps_rate, rate)
                    if self.Ps and rate - self.Ps > self.epsilon:
                        violations.append({'job': j, 'worker': w, 'reason': 'ps_capacity_exceeded', 'rate': rate, 'limit': self.Ps})
                elif ina_flag:
                    max_ina_rate = max(max_ina_rate, rate)
                    if self.Cs and rate - self.Cs > self.epsilon:
                        violations.append({'job': j, 'worker': w, 'reason': 'ina_capacity_exceeded', 'rate': rate, 'limit': self.Cs})
                else:
                    violations.append({'job': j, 'worker': w, 'reason': 'rate_without_assignment', 'rate': rate})

        status = 'ok' if not violations else 'violations'
        return {
            'status': status,
            'violations': violations,
            'max_ina_rate': max_ina_rate,
            'max_ps_rate': max_ps_rate,
        }

    def _analyze_ina_capacity(self) -> Dict[str, Any]:
        details: List[Dict[str, Any]] = []
        violations: List[Dict[str, Any]] = []
        capacity = self.Cs if self.Cs > 0 else None

        for idx in range(self.ina_candidates_num):
            load = self.switch_loads[idx] if idx < len(self.switch_loads) else 0.0
            slack = (capacity - load) if capacity is not None else None
            record = {
                'switch_id': self._candidate_id(idx),
                'load': load,
                'capacity': capacity,
                'slack': slack,
            }
            details.append(record)
            if slack is not None and slack < -self.epsilon:
                violations.append(record)

        status = 'ok' if not violations else 'violations'
        return {
            'status': status,
            'details': details,
            'violations': violations,
        }

    def _analyze_worker_rate(self) -> Dict[str, Any]:
        details: List[Dict[str, Any]] = []
        violations: List[Dict[str, Any]] = []
        for j in range(self.jobs_num):
            job_rate = self.job_rates[j] if j < len(self.job_rates) else 0.0
            derived = sum(self.worker_rate_matrix[j]) if j < len(self.worker_rate_matrix) else 0.0
            diff = job_rate - derived
            record = {
                'job': j,
                'job_rate': job_rate,
                'derived_total': derived,
                'difference': diff,
            }
            details.append(record)
            if abs(diff) > self.epsilon * max(1.0, job_rate, derived):
                violations.append(record)
        status = 'ok' if not violations else 'violations'
        return {
            'status': status,
            'details': details,
            'violations': violations,
        }

    def _analyze_egress_consistency(self) -> Dict[str, Any]:
        details: List[Dict[str, Any]] = []
        violations: List[Dict[str, Any]] = []
        for j in range(self.jobs_num):
            job_rate = self.job_rates[j] if j < len(self.job_rates) else 0.0
            switch_map = self.job_switch_loads[j] if j < len(self.job_switch_loads) else {}
            if self.job_ina_loads[j] > self.epsilon and not switch_map:
                violations.append({'job': j, 'reason': 'ina_load_without_switch'})
            for cand_idx, load in switch_map.items():
                slack = job_rate - load
                record = {
                    'job': j,
                    'switch_id': self._candidate_id(cand_idx),
                    'load': load,
                    'job_rate': job_rate,
                    'slack': slack,
                }
                details.append(record)
                if slack < -self.epsilon:
                    violations.append(record)
        status = 'ok' if not violations else 'violations'
        return {
            'status': status,
            'details': details,
            'violations': violations,
        }

    def _analyze_ina_flow_conservation(self) -> Dict[str, Any]:
        records: List[Dict[str, Any]] = []
        violations: List[Dict[str, Any]] = []
        for j in range(self.jobs_num):
            switch_sum = sum(self.job_switch_loads[j].values()) if j < len(self.job_switch_loads) else 0.0
            ina_load = self.job_ina_loads[j] if j < len(self.job_ina_loads) else 0.0
            diff = ina_load - switch_sum
            record = {
                'job': j,
                'ina_load': ina_load,
                'switch_sum': switch_sum,
                'difference': diff,
            }
            records.append(record)
            if abs(diff) > self.epsilon * max(1.0, ina_load, switch_sum):
                violations.append(record)
        status = 'ok' if not violations else 'violations'
        return {
            'status': status,
            'details': records,
            'violations': violations,
        }

    def _analyze_ps_inflow(self) -> Dict[str, Any]:
        details: List[Dict[str, Any]] = []
        violations: List[Dict[str, Any]] = []
        for j in range(self.jobs_num):
            job_rate = self.job_rates[j] if j < len(self.job_rates) else 0.0
            ps_load = self.job_ps_loads[j] if j < len(self.job_ps_loads) else 0.0
            ina_load = self.job_ina_loads[j] if j < len(self.job_ina_loads) else 0.0
            total_inflow = ps_load + ina_load
            slack = total_inflow - job_rate
            record = {
                'job': j,
                'job_rate': job_rate,
                'ps_load': ps_load,
                'ina_load': ina_load,
                'slack': slack,
            }
            details.append(record)
            if slack < -self.epsilon:
                violations.append(record)
        status = 'ok' if not violations else 'violations'
        return {
            'status': status,
            'details': details,
            'violations': violations,
        }

    def _analyze_link_capacity(self) -> Dict[str, Any]:
        ps_checks: List[Dict[str, Any]] = []
        violations: List[Dict[str, Any]] = []
        ps_capacity = self.Ps if self.Ps > 0 else None
        for j in range(self.jobs_num):
            load = self.job_ps_loads[j] if j < len(self.job_ps_loads) else 0.0
            slack = (ps_capacity - load) if ps_capacity is not None else None
            record = {
                'job': j,
                'ps_load': load,
                'capacity': ps_capacity,
                'slack': slack,
            }
            ps_checks.append(record)
            if slack is not None and slack < -self.epsilon:
                violations.append(record)

        ina_checks: List[Dict[str, Any]] = []
        capacity = self.Cs if self.Cs > 0 else None
        for idx in range(self.ina_candidates_num):
            load = self.switch_loads[idx] if idx < len(self.switch_loads) else 0.0
            slack = (capacity - load) if capacity is not None else None
            record = {
                'switch_id': self._candidate_id(idx),
                'load': load,
                'capacity': capacity,
                'slack': slack,
            }
            ina_checks.append(record)
            if slack is not None and slack < -self.epsilon:
                violations.append(record)

        status = 'ok' if not violations else 'violations'
        return {
            'status': status,
            'ps_checks': ps_checks,
            'ina_checks': ina_checks,
            'violations': violations,
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _detect_candidate_assignments(self) -> bool:
        data = self.worker_to_ina_raw
        if not isinstance(data, list):
            return False
        for job_entry in data:
            if isinstance(job_entry, (list, tuple)):
                for worker_entry in job_entry:
                    if isinstance(worker_entry, (list, tuple, dict)):
                        return True
        return False

    def _get_worker_ina_entry(self, job_index: int, worker_index: int):
        data = self.worker_to_ina_raw
        if isinstance(data, list) and job_index < len(data):
            job_entry = data[job_index]
            if isinstance(job_entry, dict):
                return job_entry.get(worker_index, 0)
            if isinstance(job_entry, (list, tuple)):
                if not job_entry:
                    return 0
                if isinstance(job_entry[0], (list, tuple, dict)):
                    if worker_index < len(job_entry):
                        return job_entry[worker_index]
                    return 0
                if worker_index < len(job_entry):
                    return job_entry[worker_index]
                return 0
            return job_entry
        return 0

    def _worker_ina_total_assignments(self, job_index: int, worker_index: int) -> float:
        entry = self._get_worker_ina_entry(job_index, worker_index)
        if isinstance(entry, (list, tuple)):
            return float(sum(1 for val in entry if isinstance(val, (int, float)) and abs(val) > self.epsilon))
        if isinstance(entry, dict):
            return float(sum(1 for val in entry.values() if isinstance(val, (int, float)) and abs(val) > self.epsilon))
        if isinstance(entry, (int, float)):
            return 1.0 if abs(entry) > self.epsilon else 0.0
        return 0.0

    def _iter_worker_candidate_assignments(self, job_index: int, worker_index: int):
        entry = self._get_worker_ina_entry(job_index, worker_index)
        if isinstance(entry, dict):
            for key, flag in entry.items():
                yield self._candidate_index_from_any(key), flag
        elif isinstance(entry, (list, tuple)):
            for idx, flag in enumerate(entry):
                yield idx, flag
        else:
            return

    def _candidate_index_from_any(self, value: Any) -> int:
        if isinstance(value, int):
            if 0 <= value < self.ina_candidates_num:
                return value
            if value in self.ina_candidates:
                return self.ina_candidates.index(value)
        return -1

    def _get_worker_ps_entry(self, job_index: int):
        data = self.worker_to_ps
        if isinstance(data, list) and job_index < len(data):
            return data[job_index]
        return data if job_index == 0 else 0

    def _worker_ps_choice(self, job_index: int, worker_index: int) -> float:
        entry = self._get_worker_ps_entry(job_index)
        value: Any = 0
        if isinstance(entry, dict):
            value = entry.get(worker_index, 0)
        elif isinstance(entry, (list, tuple)):
            if worker_index < len(entry):
                value = entry[worker_index]
        else:
            value = entry
        if isinstance(value, (int, float)):
            return 1.0 if abs(value) > self.epsilon else 0.0
        return 0.0


    def _worker_ps_raw_value(self, job_index: int, worker_index: int) -> float:
        entry = self._get_worker_ps_entry(job_index)
        value: Any = 0
        if isinstance(entry, dict):
            value = entry.get(worker_index, 0)
        elif isinstance(entry, (list, tuple)) and worker_index < len(entry):
            value = entry[worker_index]
        elif worker_index == 0:
            value = entry
        return float(value) if isinstance(value, (int, float)) else 0.0

    def _entry_has_assignment(self, entry: Any) -> bool:
        if isinstance(entry, dict):
            return any(abs(val) > self.epsilon for val in entry.values() if isinstance(val, (int, float)))
        if isinstance(entry, (list, tuple)):
            return any(abs(val) > self.epsilon for val in entry if isinstance(val, (int, float)))
        if isinstance(entry, (int, float)):
            return abs(entry) > self.epsilon
        return False

    def _prepare_worker_views(self) -> None:
        jobs_num = self.jobs_num
        align_workers = self.workers_num_align

        self.worker_ps_assignments = []
        self.worker_ina_assignments = []
        self.worker_rate_matrix = []
        self.job_ps_loads = []
        self.job_ina_loads = []
        self.job_active_workers = []
        self.job_switch_loads = []
        self.switch_loads = [0.0 for _ in range(self.ina_candidates_num)]

        for j in range(jobs_num):
            ps_row: List[float] = []
            ina_row: List[List[int]] = []
            rate_row: List[float] = [0.0 for _ in range(align_workers)]
            switch_load_map: Dict[int, float] = defaultdict(float)

            actual_workers = self.workers_num[j] if j < len(self.workers_num) else 0
            job_rate = self.job_rates[j] if j < len(self.job_rates) else 0.0
            active_worker_indices: List[int] = []

            for w in range(align_workers):
                ps_value = self._worker_ps_raw_value(j, w)
                ps_flag = ps_value > self.epsilon
                candidates = [
                    cand_idx
                    for cand_idx, flag in self._iter_worker_candidate_assignments(j, w)
                    if cand_idx >= 0 and isinstance(flag, (int, float)) and abs(flag) > self.epsilon
                ]

                ps_row.append(ps_value)
                ina_row.append(candidates)

                if w < actual_workers and (ps_flag or candidates):
                    active_worker_indices.append(w)

            active_count = len(active_worker_indices)
            per_worker_rate = job_rate / active_count if active_count else 0.0

            job_ps_total = 0.0
            job_ina_total = 0.0

            for w in active_worker_indices:
                rate_row[w] = per_worker_rate
                if ps_row[w] > self.epsilon:
                    job_ps_total += per_worker_rate
                if ina_row[w]:
                    share = per_worker_rate / len(ina_row[w]) if ina_row[w] else 0.0
                    job_ina_total += per_worker_rate
                    for cand_idx in ina_row[w]:
                        switch_load_map[cand_idx] += share
                        if 0 <= cand_idx < len(self.switch_loads):
                            self.switch_loads[cand_idx] += share

            self.worker_ps_assignments.append(ps_row)
            self.worker_ina_assignments.append(ina_row)
            self.worker_rate_matrix.append(rate_row)
            self.job_ps_loads.append(job_ps_total)
            self.job_ina_loads.append(job_ina_total)
            self.job_active_workers.append(active_count)
            self.job_switch_loads.append(dict(switch_load_map))

    def _coerce_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _validate_solution_fields(self) -> None:
        missing = [k for k in self.REQUIRED_SOLUTION_KEYS if k not in self.solution]
        if missing:
            raise ValueError(
                'ConstraintAnalyzer requires solver.solution to include '
                f"{', '.join(self.REQUIRED_SOLUTION_KEYS)}. Missing: {missing}"
            )

    def _analysis_not_available(self, reason: str) -> Dict[str, Any]:
        return {'status': 'not_available', 'reason': reason}

    def _constraint_label(self, key: str) -> str:
        return key.replace('_', ' ').title()

    def _iter_actual_worker_indices(self, job_index: int):
        workers = self.workers_num[job_index] if job_index < len(self.workers_num) else 0
        return range(workers)

    def _candidate_id(self, candidate_index: int) -> int:
        if 0 <= candidate_index < len(self.ina_candidates):
            return self.ina_candidates[candidate_index]
        return candidate_index

    def _format_report_section(self, key: str, result: Dict[str, Any]) -> List[str]:
        label = self._constraint_label(key)
        if not result:
            return [label, '  No data recorded.']

        status = result.get('status', 'n/a').upper()
        lines = [label, f"  Status : {status}"]

        if status == 'NOT_AVAILABLE':
            reason = result.get('reason', 'Not provided')
            lines.append(f"  Reason : {reason}")
            return lines

        if key == 'ina_budget':
            deployed = result.get('deployed', 0)
            budget = result.get('budget', 0)
            lines.append(f"  Deployment : {deployed}/{budget}")
            lines.append(f"  Slack      : {self._format_number(result.get('slack'))}")
            lines.append(f"  Utilization: {self._format_number(result.get('utilization'))}")
        elif key == 'worker_choice':
            tight = result.get('tight_count', 0)
            total = result.get('total_count', 0)
            lines.append(f"  Tight workers : {tight}/{total}")
            self._append_list_preview(lines, 'Violation samples', result.get('violations'), max_items=6)
        elif key == 'ina_deployment':
            lines.append(f"  Assignments : {result.get('total_assignments', 0)}")
            self._append_list_preview(lines, 'Violations', result.get('violations'), max_items=5)
        elif key == 'zero_dummy':
            lines.append(f"  Dummy workers inspected : {result.get('dummy_workers_checked', 0)}")
            self._append_list_preview(lines, 'Violations', result.get('violations'), max_items=5)
        elif key == 'bigm_coupling':
            lines.append(f"  Max INA rate : {self._format_number(result.get('max_ina_rate'))} Gbps")
            lines.append(f"  Max PS rate  : {self._format_number(result.get('max_ps_rate'))} Gbps")
            self._append_list_preview(lines, 'Violations', result.get('violations'), max_items=5)
        elif key == 'ina_capacity':
            details = result.get('details') or []
            loads = [record.get('load', 0.0) for record in details]
            if loads:
                lines.append(f"  Peak INA load : {self._format_number(max(loads))} Gbps")
            self._append_list_preview(lines, 'Violations', result.get('violations'), max_items=4)
        elif key == 'worker_rate':
            violations = result.get('violations') or []
            diffs = [abs(item.get('difference', 0.0)) for item in violations]
            if diffs:
                lines.append(f"  Max |rate gap| : {self._format_number(max(diffs))} Gbps")
            lines.append(f"  Jobs affected : {len(violations)}")
            self._append_list_preview(lines, 'Violation samples', violations, max_items=4)
        elif key == 'egress_consistency':
            self._append_list_preview(lines, 'Violations', result.get('violations'), max_items=3)
        elif key == 'ina_flow_conservation':
            self._append_list_preview(lines, 'Violations', result.get('violations'), max_items=3)
        elif key == 'ps_inflow':
            violations = result.get('violations') or []
            min_slack = min((item.get('slack', 0.0) for item in violations), default=None)
            if min_slack is not None:
                lines.append(f"  Worst slack : {self._format_number(min_slack)} Gbps")
            self._append_list_preview(lines, 'Violations', violations, max_items=4)
        elif key == 'makespan':
            lines.append(f"  Makespan : {self._format_number(result.get('makespan'))} s")
            tight_jobs = result.get('tight_jobs') or []
            slack_jobs = result.get('slack_jobs') or []
            lines.append(f"  Tight jobs / Slack jobs : {len(tight_jobs)} / {len(slack_jobs)}")
            details = result.get('details') or []
            critical = [item for item in details if item.get('is_tight')]
            if critical:
                top = critical[0]
                lines.append(f"  Critical job {top.get('job')} -> rate={self._format_number(top.get('rate'))} Gbps")
        elif key == 'link_capacity':
            ps_checks = result.get('ps_checks') or []
            ina_checks = result.get('ina_checks') or []
            lines.append(f"  Total link checks : {len(ps_checks) + len(ina_checks)}")
            self._append_list_preview(lines, 'Violations', result.get('violations'), max_items=4)
        else:
            lines.append(f"  Result : {result}")

        return lines

    def _append_list_preview(
        self,
        lines: List[str],
        title: str,
        entries: Any,
        max_items: int = 5,
        show_items: bool = True,
    ) -> None:
        if not entries:
            lines.append(f"  {title} : 0")
            return

        entries_list = list(entries)
        count = len(entries_list)
        lines.append(f"  {title} : {count}")
        if not show_items:
            return
        for entry in entries_list[:max_items]:
            lines.append(f"    - {self._stringify_entry(entry)}")
        if count > max_items:
            lines.append(f"    - ... {count - max_items} more")

    def _stringify_entry(self, entry: Any) -> str:
        if isinstance(entry, dict):
            parts = [f"{k}={entry[k]}" for k in sorted(entry.keys())]
            return ', '.join(parts)
        return str(entry)

    def _format_number(self, value: Any, precision: int = 4) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.{precision}f}"
        if value is None:
            return 'n/a'
        return str(value)

    def _section_summary_line(self, key: str, result: Dict[str, Any]) -> str:
        status = result.get('status', 'n/a')
        if status == 'not_available':
            return result.get('reason', '')
        if key == 'ina_budget':
            return f"deployed={result.get('deployed', 0)}/{result.get('budget', 0)} slack={self._format_number(result.get('slack'))}"
        if key == 'worker_choice':
            total = result.get('total_count', 0)
            tight = result.get('tight_count', 0)
            violations = len(result.get('violations') or [])
            return f"tight={tight}/{total} violations={violations}"
        if key == 'ina_deployment':
            violations = len(result.get('violations') or [])
            return f"assignments={result.get('total_assignments', 0)} violations={violations}"
        if key == 'zero_dummy':
            return f"violations={len(result.get('violations') or [])}"
        if key == 'bigm_coupling':
            return f"violations={len(result.get('violations') or [])}"
        if key == 'ina_capacity':
            return f"violations={len(result.get('violations') or [])}"
        if key == 'worker_rate':
            return f"violations={len(result.get('violations') or [])}"
        if key == 'egress_consistency':
            return f"violations={len(result.get('violations') or [])}"
        if key == 'ina_flow_conservation':
            return f"violations={len(result.get('violations') or [])}"
        if key == 'ps_inflow':
            return f"violations={len(result.get('violations') or [])}"
        if key == 'makespan':
            return f"tight_jobs={len(result.get('tight_jobs') or [])} slack_jobs={len(result.get('slack_jobs') or [])}"
        if key == 'link_capacity':
            return f"violations={len(result.get('violations') or [])}"
        return ''

    def _print_summary(self) -> None:
        print('=' * 80)
        print('CONSTRAINT SUMMARY')
        print('=' * 80)
        for key in self.CONSTRAINT_ORDER:
            label = self._constraint_label(key)
            result = self.analysis.get(key, {})
            status = result.get('status', 'n/a').upper()
            summary = self._section_summary_line(key, result)
            print(f"{label:<28} {status:<12} {summary}")
        print('=' * 80)


if __name__ == '__main__':
    # Simple smoke-test when running the module directly.
    from MMBench.problem.LLMINA.runtime.dataset import generate_dataset_job_random_fattree
    from MMBench.problem.LLMINA.runtime.topo import FatTree

    print('=' * 80)
    print('LLMINA Constraint Analysis Example')
    print('=' * 80)

    topo = FatTree(k=4, basic_band=100, hosts_num_in_fat=20, mode='fixed', k1=0.5)
    jobs_num = 3
    K = 3
    Cs = 750.0
    Ps = 200.0

    dataset = generate_dataset_job_random_fattree(
        jobs_num,
        topo.all_workers_id,
        1,
        len(topo.tors_id),
        True,
    )
    instance = list(dataset.values())[0]

    solver = ModelSolver(
        instance=instance,
        network=topo,
        K=K,
        jobs_num=jobs_num,
        Cs=Cs,
        Ps=Ps,
        topo_name='FatTree',
        solver_name='gurobi',
        time_limit=300,
        mip_gap=0.01,
        verbose=True,
    )

    solver.build_model()
    solution = solver.solve()
    if solution is None:
        raise RuntimeError('Solver failed to find a solution')

    analyzer = ConstraintAnalyzer(solver)
    analyzer.analyze_all_constraints()
    analyzer.export_report('constraint_analysis_report.txt')
