"""
通用 Solver 评估脚本

用于测试和评估任意 Solver 类的性能，包括：
1. 生成测试数据集
2. 调用 Solver 求解
3. 记录求解结果和性能指标
4. 保存评估数据

使用方法：
    1. 作为模块使用：
       from evaluate_pyomo_solver import SolverEvaluation
       evaluator = SolverEvaluation(solver_class=YourSolverClass, ...)
       
    2. 命令行使用：
       python evaluate_pyomo_solver.py
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Any, Type, Optional
import time
import json
import importlib.util

# 导入必要的模块
from .topo import FatTree, SpineLeaf
from .dataset import generate_dataset_job_random_fattree

class SolverEvaluation:
    """通用 Solver 评估器"""
    
    def __init__(
        self,
        solver_class: Type,
        topo_name: str = 'FatTree',
        ina_num_list: List[int] = [3, 5],
        jobs_num_list: List[int] = [6, 10],
        instances_num: int = 5,
        solver_name: str = 'gurobi',
        time_limit: int = 300,
        mip_gap: float = 0.01,
        verbose: bool = True
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
        self.solver_class = solver_class
        self.topo_name = topo_name
        self.ina_num_list = ina_num_list
        self.jobs_num_list = jobs_num_list
        self.instances_num = instances_num
        self.solver_name = solver_name
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.verbose = verbose
        
        # 拓扑参数
        if topo_name == 'FatTree':
            self.k = 4
            self.hosts_num_in_fat = 20
            self.tops_num = None
            self.tors_num = None
            self.hosts_num_in_tor = None
        elif topo_name == 'SpineLeaf':
            self.k = None
            self.hosts_num_in_fat = None
            self.tops_num = 5
            self.tors_num = 10
            self.hosts_num_in_tor = 20
        
        self.basic_band = 100
        self.Cs = 750.0  # INA处理容量
        self.Ps = 200.0  # PS带宽
        self.k1 = 0.5
        
        # 结果记录
        self.results = {}
    
    def setup_network(self):
        """设置网络拓扑"""
        if self.topo_name == 'FatTree':
            self.network = FatTree(
                self.k,
                self.basic_band,
                self.hosts_num_in_fat,
                'fixed',
                self.k1
            )
        elif self.topo_name == 'SpineLeaf':
            self.network = SpineLeaf(
                self.tops_num,
                self.tors_num,
                self.basic_band,
                self.hosts_num_in_tor,
                'fixed',
                self.k1
            )
        else:
            raise ValueError(f"Unknown topology: {self.topo_name}")
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"网络拓扑设置完成: {self.topo_name}")
            print(f"{'='*80}")
            print(f"  - 交换机总数: {len(self.network.all_switches_id)}")
            print(f"  - ToR交换机: {len(self.network.tors_id)}")
            print(f"  - Worker节点: {len(self.network.all_workers_id)}")
            print(f"{'='*80}\n")
    
    def generate_instances(self, jobs_num: int) -> Dict[str, dict]:
        """生成测试实例"""
        all_workers_id = self.network.all_workers_id
        
        dataset = generate_dataset_job_random_fattree(
            jobs_num,
            all_workers_id,
            self.instances_num,
            len(self.network.tors_id),
            True
        )
        
        if self.verbose:
            print(f"✓ 生成 {len(dataset)} 个测试实例 (jobs_num={jobs_num})")
        
        return dataset
    
    def solve_instance(
        self,
        instance: dict,
        K: int,
        jobs_num: int
    ) -> Dict[str, Any]:
        """
        使用指定的 Solver 类求解单个实例
        
        Args:
            instance: 问题实例
            K: INA部署预算
            jobs_num: 任务数量
        
        Returns:
            包含求解信息的字典
        """
        

        # 创建求解器实例
        start_time = time.time()
        
        solver = self.solver_class(
            instance=instance,
            network=self.network,
            K=K,
            jobs_num=jobs_num,
            Cs=self.Cs,
            Ps=self.Ps,
            topo_name=self.topo_name,
        )            
        
        # 求解
        if solver.__class__.__name__ == "PyomoSolver":
            solution = solver.solve(
                solver_name=self.solver_name,
                time_limit=self.time_limit,
                mip_gap=self.mip_gap,
                verbose=self.verbose)
        else:
            solution = solver.solve()
        solve_time = time.time() - start_time
        if self.verbose:
            print(f"✓ 求解完成: 时间 {solve_time:.2f}s")
            if solution is not None:
                print(f"  - Makespan: {solution['makespan']:.4f}")
            else:
                print(f"  - 无可行解")

        return solution

    def evaluate_configuration(
        self,
        K: int,
        jobs_num: int
    ) -> Dict[str, Any]:
        """评估单个配置（K, jobs_num）"""
        if self.verbose:
            print(f"\n{'#'*80}")
            print(f"开始评估配置: K={K}, jobs_num={jobs_num}")
            print(f"{'#'*80}\n")
        
        # 生成测试实例
        dataset = self.generate_instances(jobs_num)
        
        results = []
        
        # 求解每个实例
        for instance_idx, (name, instance) in enumerate(dataset.items(), 1):
            if self.verbose:
                print(f"\n处理实例 {instance_idx}/{len(dataset)}: {name}")
            
            # 求解
            result = self.solve_instance(instance, K, jobs_num)
            
            results.append(result)
            
        return results
    
    def evaluate_all(self) -> Dict[str, Any]:
        # 设置网络
        self.setup_network()
        
        for K in self.ina_num_list:
            for jobs_num in self.jobs_num_list:
                result = self.evaluate_configuration(K, jobs_num)
                self.results[str(K)+str(jobs_num)] = result
        
        return self.results      

def load_solver_class_from_file(file_path: str, class_name: str = "HeuristicSolver") -> Type:
    """
    从 Python 文件动态加载 Solver 类
    
    Args:
        file_path: Python 文件的路径（绝对路径或相对路径）
        class_name: 要加载的类名（默认为 "HeuristicSolver"）
    
    Returns:
        加载的类对象
    
    Raises:
        FileNotFoundError: 如果文件不存在
        AttributeError: 如果类不存在
        Exception: 其他加载错误
    """
    file_path = Path(file_path).resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"Solver 文件不存在: {file_path}")
    
    # 动态加载模块
    spec = importlib.util.spec_from_file_location("dynamic_solver_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_solver_module"] = module
    spec.loader.exec_module(module)
    
    # 获取类
    if not hasattr(module, class_name):
        raise AttributeError(f"模块中不存在类 '{class_name}': {file_path}")
    
    solver_class = getattr(module, class_name)
    
    return solver_class


def load_solver_class_from_code(code: str, class_name: str = "HeuristicSolver") -> Type:
    """
    从代码字符串动态加载 Solver 类
    
    Args:
        code: Python 代码字符串
        class_name: 要加载的类名（默认为 "HeuristicSolver"）
    
    Returns:
        加载的类对象
    
    Raises:
        AttributeError: 如果类不存在
        Exception: 代码执行错误
    """
    # 创建一个临时命名空间来执行代码
    namespace = {}
    exec(code, namespace)
    
    # 获取类
    if class_name not in namespace:
        raise AttributeError(f"代码中不存在类 '{class_name}'")
    
    solver_class = namespace[class_name]
    
    return solver_class


def evaluate_solver_from_file(
    solver_file_path: str,
    class_name: str = "ModuleSolver",
    topo_name: str = 'FatTree',
    ina_num_list: List[int] = [3],
    jobs_num_list: List[int] = [6],
    instances_num: int = 2,
    solver_name: str = 'gurobi',
    time_limit: int = 300,
    mip_gap: float = 0.01,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    从文件加载 Solver 类并进行评估的便捷函数
    
    Args:
        solver_file_path: Solver 类文件路径
        class_name: 类名（默认 "HeuristicSolver"）
        其他参数同 SolverEvaluation.__init__
        save_results: 是否保存结果到文件
        output_filename: 输出文件名（如果为 None，则自动生成）
    
    Returns:
        评估结果字典
    """
    # 加载 Solver 类
    print(f"正在从文件加载 Solver 类: {solver_file_path}")
    solver_class = load_solver_class_from_file(solver_file_path, class_name)
    print(f"✓ 成功加载类: {solver_class.__name__}\n")
    
    # 创建评估器
    evaluator = SolverEvaluation(
        solver_class=solver_class,
        topo_name=topo_name,
        ina_num_list=ina_num_list,
        jobs_num_list=jobs_num_list,
        instances_num=instances_num,
        solver_name=solver_name,
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose
    )
    
    # 运行评估
    all_results = evaluator.evaluate_all()

    return all_results


def evaluate_solver_from_code(
    solver_code: str,
    class_name: str = "HeuristicSolver",
    topo_name: str = 'FatTree',
    ina_num_list: List[int] = [3],
    jobs_num_list: List[int] = [6],
    instances_num: int = 1,
    solver_name: str = 'gurobi',
    time_limit: int = 300,
    mip_gap: float = 0.01,
    verbose: bool = True,
    save_results: bool = True,
    output_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    从代码字符串加载 Solver 类并进行评估的便捷函数
    
    Args:
        solver_code: Solver 类的完整 Python 代码字符串
        class_name: 类名（默认 "HeuristicSolver"）
        其他参数同 SolverEvaluation.__init__
        save_results: 是否保存结果到文件
        output_filename: 输出文件名（如果为 None，则自动生成）
    
    Returns:
        评估结果字典
    """
    # 加载 Solver 类
    print(f"正在从代码字符串加载 Solver 类...")
    solver_class = load_solver_class_from_code(solver_code, class_name)
    print(f"✓ 成功加载类: {solver_class.__name__}\n")
    
    # 创建评估器
    evaluator = SolverEvaluation(
        solver_class=solver_class,
        topo_name=topo_name,
        ina_num_list=ina_num_list,
        jobs_num_list=jobs_num_list,
        instances_num=instances_num,
        solver_name=solver_name,
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose
    )
    
    # 运行评估
    all_results = evaluator.evaluate_all()
    
    # 打印摘要
    evaluator.print_summary()
    
    # 保存结果
    if save_results:
        if output_filename is None:
            output_filename = f'{solver_class.__name__}_evaluation_results.json'
        evaluator.save_results(output_filename)
    
    return all_results


def main():
#    all_results = evaluate_solver_from_file(    
#         solver_file_path = "/home/hyn/LLM-MM-Agent-LLMINA-clean/output/LLMINA_20251113-151301/HeuristicSolver_final.py",
#         class_name = "HeuristicSolver",
#         topo_name='FatTree',
#         ina_num_list=[3],      # INA预算
#         jobs_num_list=[6],     # 任务数量
#         instances_num=2,       # 每个配置的实例数
#         solver_name='gurobi',
#         time_limit=300,
#         mip_gap=0.2,
#         verbose=True
#    )

   all_results = evaluate_solver_from_file(    
        solver_file_path = "/home/hyn/LLM-MM-Agent-LLMINA-clean/MMBench/problem/LLMINA/runtime/Pyomo.py",
        class_name = "PyomoSolver",
        topo_name='FatTree',
        ina_num_list=[3],      # INA预算
        jobs_num_list=[6],     # 任务数量
        instances_num=2,       # 每个配置的实例数
        solver_name='gurobi',
        time_limit=300,
        mip_gap=0.2,
        verbose=True
   )

if __name__ == "__main__":
    main()
