from __future__ import annotations
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseTemplateSolver(ABC):
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
        self.problem_data = problem_data or {}
        self.solution = None  # 缓存提取的解
           
    @abstractmethod
    def solve(self) -> Optional[Dict[str, Any]]:
        """
        求解模型并提取解
        返回值:
            - 成功时返回提取并缓存到 `self.solution` 的结果字典
            - 失败时返回 None
        """
        raise NotImplementedError
