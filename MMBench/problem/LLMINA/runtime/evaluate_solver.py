
from __future__ import annotations
from typing import Dict, List, Any, Type, Tuple, Optional
from collections import defaultdict

from .topo import FatTree, SpineLeaf
from .dataset import generate_dataset_job

class SolverEvaluation:
    """通用 Solver 评估器"""
    
    def __init__(
        self,
        solver_class: Type,
    ):
        """
        初始化评估器
        
        Args:
            solver_class: Solver 类（必须实现与 PyomoINASolver 相同的接口）
            topo_name: 拓扑类型 ('FatTree' 或 'SpineLeaf')
            ina_num_list: INA部署预算列表
            jobs_num_list: 并发任务数量列表
            instances_num: 每个配置的测试实例数量
            solver_name: Pyomo求解器名称
            time_limit: 求解时间限制（秒）
            mip_gap: MIP最优性间隙
            verbose: 是否打印详细信息
        """

        self.k = 4  # k元拓扑  # fat tree 参数

        #spine-leaf拓扑参数
        self.tops_num = 5
        self.tors_num = 10  # spine leaf 参数
        # self.tops_num = 10
        # self.tors_num = 20  # spine leaf 参数

        #共同拓扑参数
        self.topo_name = 'FatTree'
        self.hosts_num= 20  # 每个tor挂w个服务器

        self.basic_band = 100  # 底层带宽(server和tor交换机)
        self.k1 = 0.5     # 链路容量 k1×40(挂载server数)×basic_band (G)
        self.k2 = 2       # 交换机容量 k2×1600 G
        
        if self.topo_name == 'FatTree':
            self.network = FatTree(self.k, self.basic_band, self.hosts_num, 'fixed', self.k1)
        if self.topo_name == 'SpineLeaf':
            self.network = SpineLeaf(self.tops_num, self.tors_num, self.basic_band, self.hosts_num, 'fixed', self.k1)
        self.solver_class = solver_class
        

    def evaluate(self) -> Tuple[bool, str]:
        self.ina_num_list = [5]
        self.jobs_num_list = [5]
        self.Cs_list = [750.0]
        self.Ps = 200.0
        self.instance_num = 1
        
        overall_success = True
        aggregated_errors = defaultdict(list)
        total_checks = 0

        # results = []
        for ina_budget in self.ina_num_list:
            for Cs in self.Cs_list:
                for jobs_num in self.jobs_num_list:
                    dataset = generate_dataset_job(jobs_num,self.network.all_workers_id,self.instance_num,self.network.hosts_num,True)
                    
                    for name, instance in dataset.items():
                        solver = self.solver_class(
                            instance,
                            network = self.network,
                            ina_budget= ina_budget,
                            jobs_num = jobs_num,
                            Cs = Cs,
                            Ps = self.Ps,
                            topo_name = self.topo_name
                        )
                        
                        # 初始化result
                        # result = {
                        #     'solver': None,
                        #     'anwser': None,
                        # }
                        answer = solver.solve()
                        is_feasible, errors = solver.check_feasibility()
                        total_checks += 1
                        
                        if not is_feasible:
                            overall_success = False
                            for cat, msgs in errors.items():
                                # Simple deduplication by checking existence in the list
                                for msg in msgs:
                                    if msg not in aggregated_errors[cat]:
                                        aggregated_errors[cat].append(msg)
                        
                        # result['solver'] = solver
                        # result['anwser'] = answer
                        # results.append(result) 
        
        if overall_success:
             return True, f"All {total_checks} instances passed feasibility check."
        else:
             # Format string
             final_msg_lines = [f"Feasibility Failed in {total_checks} checks."]
             for cat, msgs in aggregated_errors.items():
                 final_msg_lines.append(f"[{cat}]: {len(msgs)} unique issues.")
                 # Show top 3 unique errors
                 for m in msgs[:3]:
                     final_msg_lines.append(f"    - {m}")
                 if len(msgs) > 3:
                     final_msg_lines.append(f"    - ... and {len(msgs)-3} more similar errors.")
             
             return False, "\n".join(final_msg_lines)
