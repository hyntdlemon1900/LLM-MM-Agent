"""
MILP Constraint Analyzer for LLMINA
====================================

This module analyzes the solution from PyomoINASolver to identify:
- Tight constraints (binding): constraints that are satisfied with equality
- Slack constraints (non-binding): constraints that have remaining capacity

The analyzer checks all constraint types from Pyomo.py:
1. INA placement budget constraint (ina_budget)
2. Worker aggregation choice constraints (worker_choice)
3. INA deployment constraints (ina_deployment)
4. Zero dummy workers constraints (zero_dummy_ina, zero_dummy_ps)
5. Big-M coupling constraints (bigm_ina, bigm_ps)
6. INA processing capacity constraints (ina_capacity)
7. Worker rate consistency constraints (worker_rate)
8. Aggregation egress consistency constraints (egress_consistency)
9. INA flow conservation constraints (ina_flow_conservation)
10. PS inflow sufficiency constraints (ps_inflow)
11. Makespan constraints (makespan)
12. Network link capacity constraints (link_capacity)
"""

import sys
import os
from typing import Dict, List, Tuple, Any
import numpy as np

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from MMBench.problem.LLMINA.runtime.Pyomo import PyomoINASolver


class ConstraintAnalyzer:
    """
    Analyzes MILP solution to identify tight and slack constraints.
    
    A constraint is considered:
    - Tight (binding): |LHS - RHS| < epsilon (satisfied with near-equality)
    - Slack (non-binding): LHS < RHS - epsilon (has remaining capacity)
    """
    
    def __init__(
        self, 
        solver: PyomoINASolver,
        epsilon: float = 1e-4
    ):
        """
        Initialize the constraint analyzer.
        
        Args:
            solver: PyomoINASolver instance with solved model
            epsilon: Tolerance for considering a constraint as tight
        """
        if solver.model is None or solver.results is None:
            raise ValueError("Solver must have a solved model")
        
        self.solver = solver
        self.epsilon = epsilon
        self.solution = solver.get_solution()
        
        # Extract problem data
        self.instance = solver.instance
        self.network = solver.network
        self.K = solver.K
        self.jobs_num = solver.jobs_num
        self.Cs = solver.Cs
        self.Ps = solver.Ps
        self.workers_id = solver.workers_id
        self.ps_id = solver.ps_id
        self.workers_num = solver.workers_num
        self.workers_num_align = solver.workers_num_align
        self.jobs_size = solver.jobs_size
        self.ina_candidates = solver.ina_candidates
        self.ina_candidates_num = solver.ina_candidates_num
        self.sd_id = solver.sd_id
        self.mapping = solver.mapping
        self.allPathDict = solver.allPathDict
        
        # Results storage
        self.analysis = {}
    
    def analyze_all_constraints(self) -> Dict[str, Any]:
        """
        Analyze all constraint types and categorize as tight or slack.
        
        Returns:
            Dictionary containing analysis results for all constraint types
        """
        print(f"{'='*80}")
        print(f"CONSTRAINT ANALYSIS")
        print(f"{'='*80}")
        print(f"Epsilon threshold: {self.epsilon}")
        print(f"Makespan: {self.solution['makespan']:.4f}\n")
        
        # Analyze each constraint type (按照 Pyomo.py 中的顺序)
        self.analysis['ina_budget'] = self._analyze_ina_budget()
        self.analysis['worker_choice'] = self._analyze_worker_choice()
        self.analysis['ina_deployment'] = self._analyze_ina_deployment()
        self.analysis['zero_dummy'] = self._analyze_zero_dummy()
        self.analysis['bigm_coupling'] = self._analyze_bigm_coupling()
        self.analysis['ina_capacity'] = self._analyze_ina_capacity()
        self.analysis['worker_rate'] = self._analyze_worker_rate()
        self.analysis['egress_consistency'] = self._analyze_egress_consistency()
        self.analysis['ina_flow_conservation'] = self._analyze_ina_flow_conservation()
        self.analysis['ps_inflow'] = self._analyze_ps_inflow()
        self.analysis['makespan'] = self._analyze_makespan()
        self.analysis['link_capacity'] = self._analyze_link_capacity()
        
        # Print summary
        self._print_summary()
        
        return self.analysis
    
    def _analyze_ina_budget(self) -> Dict[str, Any]:
        """Analyze INA placement budget constraint: sum(x_s) <= K"""
        num_deployed = self.solution['num_ina_deployed']
        budget = self.K
        slack = budget - num_deployed
        is_tight = slack < self.epsilon
        
        result = {
            'lhs': num_deployed,
            'rhs': budget,
            'slack': slack,
            'is_tight': is_tight,
            'utilization': num_deployed / budget if budget > 0 else 0
        }
        
        print(f"[1] INA Budget Constraint")
        print(f"    LHS (deployed): {num_deployed}")
        print(f"    RHS (budget K): {budget}")
        print(f"    Slack: {slack}")
        print(f"    Status: {'TIGHT ✓' if is_tight else 'SLACK'}")
        print(f"    Utilization: {result['utilization']*100:.1f}%\n")
        
        return result
    
    def _analyze_worker_choice(self) -> Dict[str, Any]:
        """
        Analyze worker aggregation choice constraints (Constraint 2):
        sum(y_jws[j,w,:]) + y_jwd[j,w] == 1 for all j, w (where w < workers_num[j])
        
        Each worker must select exactly one aggregation point (either INA or direct PS).
        """
        tight_count = 0
        total_count = 0
        violations = []
        
        for j in range(self.jobs_num):
            for w in range(self.workers_num[j]):  # 只检查实际的worker
                lhs = (sum(self.solution['worker_to_ina'][j][w]) + 
                       self.solution['worker_to_ps'][j][w])
                rhs = 1
                slack = abs(lhs - rhs)
                
                if slack >= self.epsilon:
                    violations.append({
                        'job': j,
                        'worker': w,
                        'lhs': lhs,
                        'slack': slack
                    })
                else:
                    tight_count += 1
                
                total_count += 1
        
        result = {
            'tight_count': tight_count,
            'total_count': total_count,
            'violations': violations,
            'all_tight': len(violations) == 0
        }
        
        print(f"[2] Worker Choice Constraints (Exclusivity)")
        print(f"    Total constraints: {total_count}")
        print(f"    Tight (satisfied): {tight_count} ({tight_count/total_count*100:.1f}%)")
        if violations:
            print(f"    ⚠ Violations: {len(violations)}")
            for v in violations[:5]:  # Show first 5
                print(f"      Job {v['job']}, Worker {v['worker']}: slack={v['slack']:.6f}")
        else:
            print(f"    Status: ALL TIGHT ✓")
        print()
        
        return result
    
    def _analyze_ina_deployment(self) -> Dict[str, Any]:
        """
        Analyze INA deployment constraints (Constraint 3):
        y_jws[j,w,s] <= x_s[s] for all j, w, s
        
        Workers can only use INA switch s if it's deployed.
        """
        violations = []
        total_count = 0
        
        for j in range(self.jobs_num):
            for w in range(self.workers_num_align):
                for s in range(self.ina_candidates_num):
                    y_jws = self.solution['worker_to_ina'][j][w][s]
                    x_s = self.solution['ina_placement'][s]
                    
                    if y_jws > x_s + self.epsilon:
                        violations.append({
                            'job': j,
                            'worker': w,
                            'switch': s,
                            'y_jws': y_jws,
                            'x_s': x_s,
                            'violation': y_jws - x_s
                        })
                    
                    total_count += 1
        
        result = {
            'total_count': total_count,
            'violations': violations,
            'satisfied': len(violations) == 0
        }
        
        print(f"[3] INA Deployment Constraints")
        print(f"    Total constraints: {total_count}")
        if violations:
            print(f"    ⚠ Violations: {len(violations)}")
            for v in violations[:5]:
                print(f"      Job {v['job']}, Worker {v['worker']}, Switch {v['switch']}: "
                      f"y_jws={v['y_jws']:.4f} > x_s={v['x_s']:.4f}")
        else:
            print(f"    Status: ALL SATISFIED ✓")
        print()
        
        return result
    
    def _analyze_zero_dummy(self) -> Dict[str, Any]:
        """
        Analyze zero dummy workers constraints (Constraint 4):
        gamma_jws[j,w,s] == 0 for w >= workers_num[j]
        gamma_jwd[j,w] == 0 for w >= workers_num[j]
        
        Dummy workers (for alignment) should have zero rates.
        """
        violations_ina = []
        violations_ps = []
        
        for j in range(self.jobs_num):
            for w in range(self.workers_num[j], self.workers_num_align):
                # Check INA rates
                for s in range(self.ina_candidates_num):
                    rate = self.solution['worker_ina_rates'][j][w][s]
                    if abs(rate) > self.epsilon:
                        violations_ina.append({
                            'job': j,
                            'worker': w,
                            'switch': s,
                            'rate': rate
                        })
                
                # Check PS rates
                rate = self.solution['worker_ps_rates'][j][w]
                if abs(rate) > self.epsilon:
                    violations_ps.append({
                        'job': j,
                        'worker': w,
                        'rate': rate
                    })
        
        total_dummy = sum(self.workers_num_align - self.workers_num[j] for j in range(self.jobs_num))
        
        result = {
            'total_dummy_workers': total_dummy,
            'violations_ina': violations_ina,
            'violations_ps': violations_ps,
            'satisfied': len(violations_ina) == 0 and len(violations_ps) == 0
        }
        
        print(f"[4] Zero Dummy Workers Constraints")
        print(f"    Total dummy workers: {total_dummy}")
        if violations_ina or violations_ps:
            print(f"    ⚠ INA rate violations: {len(violations_ina)}")
            print(f"    ⚠ PS rate violations: {len(violations_ps)}")
            for v in violations_ina[:3]:
                print(f"      Job {v['job']}, Dummy Worker {v['worker']}, Switch {v['switch']}: "
                      f"rate={v['rate']:.6f} (should be 0)")
        else:
            print(f"    Status: ALL SATISFIED ✓")
        print()
        
        return result
    
    def _analyze_bigm_coupling(self) -> Dict[str, Any]:
        """
        Analyze Big-M coupling constraints (Constraint 5-6):
        gamma_jws[j,w,s] <= M_ina * y_jws[j,w,s]  (M_ina = Cs)
        gamma_jwd[j,w] <= M_ps * y_jwd[j,w]  (M_ps = Ps)
        
        Rates should be zero when assignment variable is 0.
        """
        violations_ina = []
        violations_ps = []
        
        M_ina = self.Cs
        M_ps = self.Ps
        
        for j in range(self.jobs_num):
            for w in range(self.workers_num_align):
                # Check INA coupling
                for s in range(self.ina_candidates_num):
                    rate = self.solution['worker_ina_rates'][j][w][s]
                    y = self.solution['worker_to_ina'][j][w][s]
                    rhs = M_ina * y
                    
                    if rate > rhs + self.epsilon:
                        violations_ina.append({
                            'job': j,
                            'worker': w,
                            'switch': s,
                            'rate': rate,
                            'y': y,
                            'violation': rate - rhs
                        })
                
                # Check PS coupling
                rate = self.solution['worker_ps_rates'][j][w]
                y = self.solution['worker_to_ps'][j][w]
                rhs = M_ps * y
                
                if rate > rhs + self.epsilon:
                    violations_ps.append({
                        'job': j,
                        'worker': w,
                        'rate': rate,
                        'y': y,
                        'violation': rate - rhs
                    })
        
        result = {
            'violations_ina': violations_ina,
            'violations_ps': violations_ps,
            'satisfied': len(violations_ina) == 0 and len(violations_ps) == 0
        }
        
        print(f"[5] Big-M Coupling Constraints")
        if violations_ina or violations_ps:
            print(f"    ⚠ INA coupling violations: {len(violations_ina)}")
            print(f"    ⚠ PS coupling violations: {len(violations_ps)}")
        else:
            print(f"    Status: ALL SATISFIED ✓")
        print()
        
        return result
    
    def _analyze_ina_capacity(self) -> Dict[str, Any]:
        """
        Analyze INA switch processing capacity constraints (Constraint 7):
        sum_j sum_w gamma_jws[j,w,s] <= Cs * x_s[s]
        
        Total ingress to each deployed INA should not exceed its capacity.
        """
        tight_switches = []
        slack_switches = []
        
        for s in range(self.ina_candidates_num):
            if self.solution['ina_placement'][s] == 0:
                continue  # Skip non-deployed switches
            
            # Calculate total ingress to this INA switch
            total_ingress = 0
            for j in range(self.jobs_num):
                for w in range(self.workers_num[j]):  # 只计算实际worker
                    total_ingress += self.solution['worker_ina_rates'][j][w][s]
            
            capacity = self.Cs * self.solution['ina_placement'][s]
            slack = capacity - total_ingress
            utilization = total_ingress / capacity if capacity > 0 else 0
            
            switch_info = {
                'switch_id': self.ina_candidates[s],
                'ingress': total_ingress,
                'capacity': capacity,
                'slack': slack,
                'utilization': utilization,
                'is_tight': slack < self.epsilon
            }
            
            if switch_info['is_tight']:
                tight_switches.append(switch_info)
            else:
                slack_switches.append(switch_info)
        
        result = {
            'tight_switches': tight_switches,
            'slack_switches': slack_switches,
            'num_tight': len(tight_switches),
            'num_slack': len(slack_switches),
            'total_deployed': len(tight_switches) + len(slack_switches)
        }
        
        print(f"[6] INA Processing Capacity Constraints")
        print(f"    Deployed switches: {result['total_deployed']}")
        print(f"    Tight constraints: {result['num_tight']}")
        print(f"    Slack constraints: {result['num_slack']}")
        
        if tight_switches:
            print(f"    Tight switches (utilizing ≥{(1-self.epsilon/self.Cs)*100:.1f}% capacity):")
            for sw in tight_switches:
                print(f"      Switch {sw['switch_id']}: {sw['ingress']:.2f}/{sw['capacity']:.2f} Gbps ({sw['utilization']*100:.1f}%) ✓")
        
        if slack_switches:
            print(f"    Slack switches:")
            for sw in slack_switches[:5]:  # Show first 5
                print(f"      Switch {sw['switch_id']}: {sw['ingress']:.2f}/{sw['capacity']:.2f} Gbps ({sw['utilization']*100:.1f}%), slack={sw['slack']:.2f}")
        print()
        
        return result
    
    def _analyze_worker_rate(self) -> Dict[str, Any]:
        """
        Analyze worker rate consistency constraints (Constraint 8):
        sum_s gamma_jws[j,w,s] + gamma_jwd[j,w] >= gamma_j[j]
        
        Each worker's total rate must meet the job's effective rate.
        """
        tight_count = 0
        slack_count = 0
        violations = []
        details = []
        
        for j in range(self.jobs_num):
            job_rate = self.solution['job_rates'][j]
            
            for w in range(self.workers_num[j]):  # 只检查实际worker
                worker_rate = (sum(self.solution['worker_ina_rates'][j][w]) + 
                              self.solution['worker_ps_rates'][j][w])
                slack = worker_rate - job_rate
                is_tight = slack < self.epsilon
                
                if slack < -self.epsilon:  # 违反约束
                    violations.append({
                        'job': j,
                        'worker': w,
                        'worker_rate': worker_rate,
                        'job_rate': job_rate,
                        'violation': -slack
                    })
                
                details.append({
                    'job': j,
                    'worker': w,
                    'worker_rate': worker_rate,
                    'job_rate': job_rate,
                    'slack': slack,
                    'is_tight': is_tight
                })
                
                if is_tight:
                    tight_count += 1
                else:
                    slack_count += 1
        
        result = {
            'tight_count': tight_count,
            'slack_count': slack_count,
            'total_count': tight_count + slack_count,
            'violations': violations,
            'details': details
        }
        
        print(f"[7] Worker Rate Consistency Constraints")
        print(f"    Total constraints: {result['total_count']}")
        print(f"    Tight: {tight_count} ({tight_count/result['total_count']*100:.1f}%)")
        print(f"    Slack: {slack_count} ({slack_count/result['total_count']*100:.1f}%)")
        
        if violations:
            print(f"    ⚠ Violations: {len(violations)}")
            for v in violations[:3]:
                print(f"      Job {v['job']}, Worker {v['worker']}: {v['worker_rate']:.2f} < {v['job_rate']:.2f} (violation={v['violation']:.4f})")
        else:
            # Show some tight examples
            tight_examples = [d for d in details if d['is_tight']][:3]
            if tight_examples:
                print(f"    Tight examples:")
                for ex in tight_examples:
                    print(f"      Job {ex['job']}, Worker {ex['worker']}: {ex['worker_rate']:.2f} ≥ {ex['job_rate']:.2f} (slack={ex['slack']:.4f}) ✓")
        print()
        
        return result
    
    def _analyze_egress_consistency(self) -> Dict[str, Any]:
        """
        Analyze aggregation egress consistency constraints (Constraint 9):
        gamma_js[j,s] >= gamma_jws[j,w,s] for all j, w, s
        
        INA egress rate must not be slower than any worker's ingress rate.
        """
        violations = []
        total_count = 0
        
        for j in range(self.jobs_num):
            for w in range(self.workers_num[j]):  # 只检查实际worker
                for s in range(self.ina_candidates_num):
                    worker_rate = self.solution['worker_ina_rates'][j][w][s]
                    egress_rate = self.solution['ina_egress_rates'][j][s]
                    
                    if worker_rate > egress_rate + self.epsilon:
                        violations.append({
                            'job': j,
                            'worker': w,
                            'switch': s,
                            'worker_rate': worker_rate,
                            'egress_rate': egress_rate,
                            'violation': worker_rate - egress_rate
                        })
                    
                    total_count += 1
        
        result = {
            'total_count': total_count,
            'violations': violations,
            'satisfied': len(violations) == 0
        }
        
        print(f"[8] Aggregation Egress Consistency Constraints")
        print(f"    Total constraints: {total_count}")
        if violations:
            print(f"    ⚠ Violations: {len(violations)}")
            for v in violations[:5]:
                print(f"      Job {v['job']}, Worker {v['worker']}, Switch {v['switch']}: "
                      f"worker_rate={v['worker_rate']:.2f} > egress={v['egress_rate']:.2f}")
        else:
            print(f"    Status: ALL SATISFIED ✓")
        print()
        
        return result
    
    def _analyze_ina_flow_conservation(self) -> Dict[str, Any]:
        """
        Analyze INA flow conservation constraints (Constraint 9.1):
        gamma_js[j,s] <= sum_w gamma_jws[j,w,s]
        
        INA egress rate cannot exceed sum of ingress rates.
        """
        violations = []
        details = []
        
        for j in range(self.jobs_num):
            for s in range(self.ina_candidates_num):
                egress_rate = self.solution['ina_egress_rates'][j][s]
                total_ingress = sum(self.solution['worker_ina_rates'][j][w][s] 
                                   for w in range(self.workers_num[j]))
                
                slack = total_ingress - egress_rate
                
                if slack < -self.epsilon:  # 违反约束
                    violations.append({
                        'job': j,
                        'switch': s,
                        'egress_rate': egress_rate,
                        'total_ingress': total_ingress,
                        'violation': -slack
                    })
                
                details.append({
                    'job': j,
                    'switch': s,
                    'egress_rate': egress_rate,
                    'total_ingress': total_ingress,
                    'slack': slack,
                    'is_tight': abs(slack) < self.epsilon
                })
        
        result = {
            'total_count': len(details),
            'violations': violations,
            'details': details,
            'satisfied': len(violations) == 0
        }
        
        print(f"[9] INA Flow Conservation Constraints")
        print(f"    Total constraints: {result['total_count']}")
        if violations:
            print(f"    ⚠ Violations: {len(violations)}")
            for v in violations[:5]:
                print(f"      Job {v['job']}, Switch {v['switch']}: "
                      f"egress={v['egress_rate']:.2f} > ingress={v['total_ingress']:.2f}")
        else:
            print(f"    Status: ALL SATISFIED ✓")
        print()
        
        return result
    
    def _analyze_ps_inflow(self) -> Dict[str, Any]:
        """
        Analyze PS inflow sufficiency constraints (Constraint 9.5):
        gamma_j[j] <= sum_s gamma_js[j,s] + sum_w gamma_jwd[j,w]
        
        Job rate cannot exceed total PS inflow.
        """
        violations = []
        details = []
        
        for j in range(self.jobs_num):
            job_rate = self.solution['job_rates'][j]
            ina_inflow = sum(self.solution['ina_egress_rates'][j])
            direct_inflow = sum(self.solution['worker_ps_rates'][j][w] 
                               for w in range(self.workers_num[j]))
            total_inflow = ina_inflow + direct_inflow
            
            slack = total_inflow - job_rate
            
            if slack < -self.epsilon:  # 违反约束
                violations.append({
                    'job': j,
                    'job_rate': job_rate,
                    'total_inflow': total_inflow,
                    'ina_inflow': ina_inflow,
                    'direct_inflow': direct_inflow,
                    'violation': -slack
                })
            
            details.append({
                'job': j,
                'job_rate': job_rate,
                'total_inflow': total_inflow,
                'ina_inflow': ina_inflow,
                'direct_inflow': direct_inflow,
                'slack': slack,
                'is_tight': abs(slack) < self.epsilon
            })
        
        result = {
            'total_count': len(details),
            'violations': violations,
            'details': details,
            'satisfied': len(violations) == 0
        }
        
        print(f"[10] PS Inflow Sufficiency Constraints")
        print(f"    Total constraints: {result['total_count']}")
        if violations:
            print(f"    ⚠ Violations: {len(violations)}")
            for v in violations[:5]:
                print(f"      Job {v['job']}: job_rate={v['job_rate']:.2f} > "
                      f"total_inflow={v['total_inflow']:.2f} (INA={v['ina_inflow']:.2f}, Direct={v['direct_inflow']:.2f})")
        else:
            tight_examples = [d for d in details if d['is_tight']]
            if tight_examples:
                print(f"    Tight examples:")
                for ex in tight_examples[:3]:
                    print(f"      Job {ex['job']}: job_rate={ex['job_rate']:.2f} ≈ "
                          f"total_inflow={ex['total_inflow']:.2f} ✓")
            print(f"    Status: ALL SATISFIED")
        print()
        
        return result
    
    def _analyze_makespan(self) -> Dict[str, Any]:
        """
        Analyze makespan constraints (Constraint 10):
        alpha <= gamma_j[j] / m_j for all j  (i.e., makespan >= m_j / gamma_j)
        
        Find which jobs are limiting the makespan.
        """
        tight_jobs = []
        slack_jobs = []
        
        alpha = self.solution['alpha']
        makespan = self.solution['makespan']
        
        for j in range(self.jobs_num):
            job_rate = self.solution['job_rates'][j]
            job_size = self.jobs_size[j]
            job_time = job_size / job_rate if job_rate > 1e-9 else float('inf')
            
            # alpha <= gamma_j / m_j  <=>  job_time >= makespan
            slack = job_time - makespan
            is_tight = abs(slack) < self.epsilon
            
            job_info = {
                'job': j,
                'rate': job_rate,
                'size': job_size,
                'time': job_time,
                'slack': slack,
                'is_tight': is_tight
            }
            
            if is_tight:
                tight_jobs.append(job_info)
            else:
                slack_jobs.append(job_info)
        
        result = {
            'makespan': makespan,
            'alpha': alpha,
            'tight_jobs': tight_jobs,
            'slack_jobs': slack_jobs,
            'num_tight': len(tight_jobs),
            'num_slack': len(slack_jobs)
        }
        
        print(f"[11] Makespan Constraints")
        print(f"    Makespan: {makespan:.4f}")
        print(f"    Alpha (inverse): {alpha:.6f}")
        print(f"    Jobs limiting makespan (tight): {result['num_tight']}")
        print(f"    Jobs with slack: {result['num_slack']}")
        
        if tight_jobs:
            print(f"    Bottleneck jobs:")
            for job in tight_jobs:
                print(f"      Job {job['job']}: time={job['time']:.4f} (rate={job['rate']:.2f} Gbps, size={job['size']:.1f} GB) ✓")
        
        if slack_jobs:
            print(f"    Jobs with slack:")
            for job in slack_jobs[:5]:
                print(f"      Job {job['job']}: time={job['time']:.4f}, slack={job['slack']:.4f}")
        print()
        
        return result
    
    def _analyze_link_capacity(self) -> Dict[str, Any]:
        """
        Analyze network link capacity constraints (Constraint 11):
        sum of all flows on link <= link_capacity
        
        Check both regular links and PS last-hop links.
        """
        tight_links = []
        slack_links = []
        
        for link_id, link_info in self.mapping.items():
            edge = link_info[0]
            base_capacity = link_info[1]
            
            # Calculate total load on this link
            total_load = 0
            flow_details = []
            
            # Determine actual capacity (PS bandwidth for last-hop links)
            capacity = base_capacity
            for j in range(self.jobs_num):
                if (edge == (self.sd_id[j], self.ps_id[j]) or 
                    edge == (self.ps_id[j], self.sd_id[j])):
                    capacity = self.Ps
                    break
            
            # Accumulate all flows on this link
            for j in range(self.jobs_num):
                # Worker-to-INA flows
                for w in range(self.workers_num[j]):
                    for s in range(self.ina_candidates_num):
                        rate = self.solution['worker_ina_rates'][j][w][s]
                        if rate > 1e-6:
                            src = self.workers_id[j][w]
                            dst = self.ina_candidates[s]
                            if self._flow_uses_link(src, dst, edge):
                                total_load += rate
                                flow_details.append(f"W{j}.{w}→INA{s}: {rate:.2f}")
                
                # Worker-to-PS flows
                for w in range(self.workers_num[j]):
                    rate = self.solution['worker_ps_rates'][j][w]
                    if rate > 1e-6:
                        src = self.workers_id[j][w]
                        dst = self.ps_id[j]
                        if self._flow_uses_link(src, dst, edge):
                            total_load += rate
                            flow_details.append(f"W{j}.{w}→PS{j}: {rate:.2f}")
                
                # INA-to-PS flows
                for s in range(self.ina_candidates_num):
                    rate = self.solution['ina_egress_rates'][j][s]
                    if rate > 1e-6:
                        src = self.ina_candidates[s]
                        dst = self.ps_id[j]
                        if self._flow_uses_link(src, dst, edge):
                            total_load += rate
                            flow_details.append(f"INA{s}→PS{j}: {rate:.2f}")
            
            slack = capacity - total_load
            utilization = total_load / capacity if capacity > 0 else 0
            is_tight = slack < self.epsilon
            
            link_data = {
                'link_id': link_id,
                'edge': edge,
                'load': total_load,
                'capacity': capacity,
                'slack': slack,
                'utilization': utilization,
                'is_tight': is_tight,
                'flows': flow_details,
                'is_ps_link': capacity == self.Ps
            }
            
            if is_tight:
                tight_links.append(link_data)
            else:
                slack_links.append(link_data)
        
        # Sort by utilization
        tight_links.sort(key=lambda x: x['utilization'], reverse=True)
        slack_links.sort(key=lambda x: x['utilization'], reverse=True)
        
        result = {
            'tight_links': tight_links,
            'slack_links': slack_links,
            'num_tight': len(tight_links),
            'num_slack': len(slack_links),
            'total_links': len(tight_links) + len(slack_links)
        }
        
        print(f"[12] Network Link Capacity Constraints")
        print(f"    Total links: {result['total_links']}")
        print(f"    Tight: {result['num_tight']} ({result['num_tight']/result['total_links']*100:.1f}%)")
        print(f"    Slack: {result['num_slack']} ({result['num_slack']/result['total_links']*100:.1f}%)")
        
        if tight_links:
            print(f"    Top tight links:")
            for link in tight_links[:5]:
                link_type = " [PS-link]" if link['is_ps_link'] else ""
                print(f"      Link {link['edge']}{link_type}: {link['load']:.2f}/{link['capacity']:.2f} Gbps ({link['utilization']*100:.1f}%) ✓")
                if len(link['flows']) <= 3:
                    for flow in link['flows']:
                        print(f"        - {flow}")
        
        if slack_links:
            print(f"    Most utilized slack links:")
            for link in slack_links[:3]:
                link_type = " [PS-link]" if link['is_ps_link'] else ""
                print(f"      Link {link['edge']}{link_type}: {link['load']:.2f}/{link['capacity']:.2f} Gbps ({link['utilization']*100:.1f}%), slack={link['slack']:.2f}")
        print()
        
        return result
    
    def _flow_uses_link(self, src: int, dst: int, edge: Tuple[int, int]) -> bool:
        """Check if flow from src to dst uses the given edge."""
        path = self.allPathDict[src][dst]
        links = [(path[i], path[i+1]) for i in range(len(path)-1)]
        return edge in links or edge[::-1] in links
    
    def _print_summary(self):
        """Print overall summary of constraint analysis."""
        print(f"{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        
        total_tight = 0
        total_constraints = 0
        total_violations = 0
        
        # 1. INA budget (1 constraint)
        total_constraints += 1
        if self.analysis['ina_budget']['is_tight']:
            total_tight += 1
        
        # 2. Worker choice (equality constraints)
        wc = self.analysis['worker_choice']
        total_constraints += wc['total_count']
        total_tight += wc['tight_count']
        total_violations += len(wc['violations'])
        
        # 3. INA deployment (inequality constraints)
        id_analysis = self.analysis['ina_deployment']
        total_constraints += id_analysis['total_count']
        total_violations += len(id_analysis['violations'])
        if id_analysis['satisfied']:
            total_tight += id_analysis['total_count']
        
        # 4. Zero dummy workers
        zd = self.analysis['zero_dummy']
        if zd['total_dummy_workers'] > 0:
            total_constraints += zd['total_dummy_workers'] * (self.ina_candidates_num + 1)
            total_violations += len(zd['violations_ina']) + len(zd['violations_ps'])
        
        # 5. Big-M coupling
        bm = self.analysis['bigm_coupling']
        total_constraints += self.jobs_num * self.workers_num_align * (self.ina_candidates_num + 1)
        total_violations += len(bm['violations_ina']) + len(bm['violations_ps'])
        
        # 6. INA capacity
        ic = self.analysis['ina_capacity']
        total_constraints += ic['total_deployed']
        total_tight += ic['num_tight']
        
        # 7. Worker rate
        wr = self.analysis['worker_rate']
        total_constraints += wr['total_count']
        total_tight += wr['tight_count']
        total_violations += len(wr['violations'])
        
        # 8. Egress consistency
        ec = self.analysis['egress_consistency']
        total_constraints += ec['total_count']
        total_violations += len(ec['violations'])
        
        # 9. INA flow conservation
        fc = self.analysis['ina_flow_conservation']
        total_constraints += fc['total_count']
        total_violations += len(fc['violations'])
        
        # 10. PS inflow
        pi = self.analysis['ps_inflow']
        total_constraints += pi['total_count']
        total_violations += len(pi['violations'])
        
        # 11. Makespan
        mk = self.analysis['makespan']
        total_constraints += self.jobs_num
        total_tight += mk['num_tight']
        
        # 12. Link capacity
        lc = self.analysis['link_capacity']
        total_constraints += lc['total_links']
        total_tight += lc['num_tight']
        
        print(f"Total constraints checked: {total_constraints}")
        print(f"Tight constraints: {total_tight} ({total_tight/total_constraints*100:.1f}%)")
        print(f"Slack constraints: {total_constraints - total_tight - total_violations} "
              f"({(total_constraints-total_tight-total_violations)/total_constraints*100:.1f}%)")
        
        if total_violations > 0:
            print(f"\n⚠ VIOLATIONS DETECTED: {total_violations}")
            print(f"  Solution may be infeasible or numerical errors present!")
        else:
            print(f"\n✓ All constraints satisfied (no violations)")
        
        print(f"{'='*80}\n")
    
    def export_report(self, filepath: str = "constraint_analysis_report.txt"):
        """Export detailed analysis report to a text file."""
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LLMINA MILP CONSTRAINT ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Makespan: {self.solution['makespan']:.4f}\n")
            f.write(f"INA switches deployed: {self.solution['num_ina_deployed']}/{self.K}\n")
            f.write(f"Epsilon threshold: {self.epsilon}\n\n")
            
            # Detailed constraints
            for key, value in self.analysis.items():
                f.write(f"\n{key.upper().replace('_', ' ')}:\n")
                f.write(str(value) + "\n")
        
        print(f"✓ Detailed report exported to: {filepath}")


if __name__ == "__main__":
    """
    Example usage of ConstraintAnalyzer with PyomoINASolver.
    """
    import sys
    import os
    import time
    
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from MMBench.problem.LLMINA.runtime.topo import FatTree
    from MMBench.problem.LLMINA.runtime.dataset import generate_dataset_job_random_fattree
    from MMBench.problem.LLMINA.runtime.Pyomo import PyomoINASolver
    
    print("="*80)
    print("LLMINA Constraint Analysis Example")
    print("="*80)
    
    # Create network topology
    print("\n[1] Creating FatTree topology...")
    k = 4
    basic_band = 100
    hosts_num_in_fat = 20
    k1 = 0.5
    
    network = FatTree(k, basic_band, hosts_num_in_fat, 'fixed', k1)
    all_workers_id = network.all_workers_id
    
    print(f"✓ Topology created: {len(all_workers_id)} workers")
    
    # Generate instance
    print("\n[2] Generating problem instance...")
    jobs_num = 3
    K = 3
    Cs = 750.0
    Ps = 200.0
    
    dataset = generate_dataset_job_random_fattree(
        jobs_num, 
        all_workers_id, 
        1,
        len(network.tors_id),
        True
    )
    
    instance = list(dataset.values())[0]
    print(f"✓ Instance generated: {jobs_num} jobs")
    
    # Solve with MILP
    print("\n[3] Solving MILP...")
    solver = PyomoINASolver(
        instance=instance,
        network=network,
        K=K,
        jobs_num=jobs_num,
        Cs=Cs,
        Ps=Ps,
        topo_name='FatTree',
        solver_name='gurobi',
        time_limit=300,
        mip_gap=0.01,
        verbose=True
    )
    
    solver.build_model()
    solution = solver.solve()

    if solution is None:
        print("✗ Solver failed to find solution")
        exit(1)
    
    print(f"✓ Optimal solution found")
    print(f"  Makespan: {solver.get_makespan():.4f}")
    
    # Analyze constraints
    print("\n[4] Analyzing constraints...")
    analyzer = ConstraintAnalyzer(solver, epsilon=1e-4)
    analysis = analyzer.analyze_all_constraints()
    
    # Export report
    analyzer.export_report("constraint_analysis_report.txt")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
