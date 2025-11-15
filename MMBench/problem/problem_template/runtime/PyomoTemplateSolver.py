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
from typing import Dict, Any, Optional
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
from .BaseTemplateSolver import BaseTemplateSolver
from abc import ABC, abstractmethod
import inspect

class PyomoTemplateSolver(BaseTemplateSolver):
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
               
    def build_model(self):
        """
        构建Pyomo优化模型（抽象方法，子类必须重写）
        
        子类应在此方法中：
        1. 调用 self._preprocess_data() 进行数据预处理（可选）
        2. 创建 pyo.ConcreteModel 实例
        3. 定义集合（Sets）、变量（Vars）、目标函数（Objective）、约束（Constraints）
        4. 将模型赋值给 self.model
        
        示例：
            model = pyo.ConcreteModel()
            model.I = pyo.RangeSet(0, n-1)
            model.x = pyo.Var(model.I, domain=pyo.Binary)
            model.obj = pyo.Objective(expr=..., sense=pyo.minimize)
            model.constraint = pyo.Constraint(model.I, rule=...)
            self.model = model
        """
        raise NotImplementedError("子类必须实现 build_model() 方法")
    
    def solve_pyomo(
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

        返回值:
            - 成功时返回提取并缓存到 `self.solution` 的结果字典
            - 失败时返回 None
        """        
        self.solver_name = solver_name
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.verbose = verbose

        self.build_model()
        
        solver = SolverFactory(self.solver_name)
        if not solver.available():
            raise RuntimeError(f"求解器 '{self.solver_name}' 不可用")
        
        self._configure_solver_options(solver)
         
        try:
            self.results = solver.solve(self.model,tee=self.verbose)
        except Exception as e:
            self.solution = None
            return None
      
        success = self._check_solution_status()

        # 自动提取解并返回
        if not success:
            self.solution = None
            return None

        try:
            self.solution = self.get_solution()
            return self.solution
              
        except Exception as e:
            self.solution = None
            return None
        
    @abstractmethod
    def get_solution(self) -> Optional[Dict[str, Any]]:
        """
        获取已缓存的解
        
        返回值:
            - 成功时返回缓存的结果字典
            - 失败时返回 None
        """
        return self.solution

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
            return False
        
        term_cond = self.results.solver.termination_condition
        
        if term_cond == TerminationCondition.optimal:
            return True
        
        if term_cond in [TerminationCondition.feasible, TerminationCondition.maxTimeLimit,
                        TerminationCondition.maxIterations, TerminationCondition.maxEvaluations]:
            try:
                obj_value = pyo.value(self.model.objective)
                if obj_value is not None and self.verbose:
                    print(f"⚠ 可行解（非最优）: 目标值={obj_value:.6f}")
                return obj_value is not None
            except:
                pass
        
        return False
    
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

        if hasattr(self, 'solve_heuristic') and callable(getattr(self, 'solve_heuristic')):
            return self._call_with_filtered_kwargs(self.solve_heuristic, **solver_kwargs)
        else:
            return self._call_with_filtered_kwargs(self.solve_pyomo, **solver_kwargs)
