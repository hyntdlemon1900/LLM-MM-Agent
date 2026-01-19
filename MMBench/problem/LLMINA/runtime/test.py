# from .ModelSolver import ModelSolver
# from .topo import FatTree, SpineLeaf
# from .dataset import generate_dataset_job
# import random
# network = FatTree(4, 100, 20, 'fixed', 0.5)
    
# all_workers_id = network.all_workers_id # server id列表
# random.seed(42)
# dataset = generate_dataset_job(
#     jobs_num=2,
#     all_workers_id=all_workers_id,
#     num_instances=1,
#     num_edge_switches=20,
#     sequential_allocation=True
# )
# print(dataset)
# solver = ModelSolver(
#     instance=dataset['0'],
#     network=network,
#     ina_budget5,
#     jobs_num=2,
#     Cs=750.0,
#     Ps=200.0,
#     topo_name='FatTree'
# )
# solver.build_model()
# solution = solver.solve_with_pyomo(
#     solver_name='gurobi',
#     time_limit=300,
#     mip_gap=0.01,
#     verbose=True
# )

# makespan = solver.model.alpha.value
# print("Makespan (1/alpha):", 1.0 / makespan if makespan else None)
# print("INA Placement Switches:", solution["ina_placement_switches"])
# print("Worker Aggregation IDs:", solution["worker_agg_id"])
# ina_placement_switches = solution["ina_placement_switches"]
# worer_agg_id = solution["worker_agg_id"]
# makespan = solver.get_makespan(ina_placement_switches, worer_agg_id)
# print("Makespan:", makespan)



# from pathlib import Path
# import importlib.util
# from MMBench.problem.LLMINA.runtime.solutionanalyzer import SolutionAnalyzer
# from typing import Dict, List, Any, Type, Optional
# def load_solver_class_from_file(file_path: str, class_name: str = "ModelSolver") -> Type:
#     file_path = Path(file_path).resolve()
    
#     if not file_path.exists():
#         raise FileNotFoundError(f"Solver 文件不存在: {file_path}")
    
#     # 动态加载模块
#     spec = importlib.util.spec_from_file_location("dynamic_solver_module", file_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)

#     solver_class = getattr(module, class_name, None)
#     if solver_class is None:
#         raise AttributeError(f"模块中不存在类 '{class_name}': {file_path}")

#     return solver_class
# solver_class = load_solver_class_from_file('/home/hyn/LLM-MM-Agent-LLMINA-clean/output/LLMINA_20251205-113010/algorithm/5/ModelSolver_final.py')
# analyzer = SolutionAnalyzer(solver_class=solver_class)
# quality_report = analyzer.run_batch_evaluation()
# print(quality_report)



from MMBench.problem.LLMINA.runtime.BaseSolve import RelaxSolve
from MMBench.problem.LLMINA.runtime.dataset import generate_dataset_job
from MMBench.problem.LLMINA.runtime.topo import FatTree, SpineLeaf
ina_num = 5
jobs_num = 5
Cs = 750.0
instance_num = 1
k = 4  # FatTree 参数
tops_num = 5
tors_num = 10 
hosts_num = 20
basic_band = 100
k1 = 0.5 
k2 = 2

topo_name = 'FatTree' # 或 'SpineLeaf'

# 初始化物理网络对象
if topo_name == 'FatTree':
    network = FatTree(k, basic_band, hosts_num, 'fixed', k1)
elif topo_name == 'SpineLeaf':
    network = SpineLeaf(tops_num, tors_num, basic_band, hosts_num, 'fixed', k1)

dataset = generate_dataset_job(jobs_num, network.all_workers_id, instance_num, network.hosts_num, True)
for exp_id, name in enumerate(dataset):  
    instance = dataset[name]
    base_solver = RelaxSolve(ina_num, jobs_num, Cs, network, instance)
    Jct = base_solver.base_solve()
    print(f"Instance {exp_id} - Makespan: {Jct}")