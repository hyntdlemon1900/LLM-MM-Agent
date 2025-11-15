"""
模板生成器 - 从Pyomo求解器生成LLM求解器模板

核心功能：
1. 接收Pyomo.py文件内容作为输入
2. 使用LLM分析Pyomo求解器的接口规范
3. 生成简洁的llm_solver函数模板
"""

import os
from typing import Optional
from MMAgent.prompt.prompt_template import TEMPLATE_GENERATION_PROMPT

class TemplateGenerator:
    """
    代码模板生成器
    
    从Pyomo求解器代码生成LLM求解器模板
    """
    
    def __init__(self, llm, problem_dir: str):
        """
        初始化模板生成器
        
        Args:
            llm: 语言模型实例
            problem_dir: 问题目录路径（用于加载Pyomo.py）
        """
        self.llm = llm
        self.problem_dir = problem_dir
        
        # 加载Pyomo.py文件
        pyomo_path = os.path.join(problem_dir, 'runtime', 'Pyomo.py')
        if not os.path.exists(pyomo_path):
            raise FileNotFoundError(f"Pyomo.py not found at {pyomo_path}")
        
        with open(pyomo_path, 'r', encoding='utf-8') as f:
            self.pyomo_code = f.read()
    
    def generate_template(self) -> str:
        """
        生成llm_solver函数模板
        
        使用LLM分析Pyomo求解器，提取接口规范并生成模板
        
        Returns:
            生成的模板代码字符串
        """
        
        
        # 构建prompt
        prompt = TEMPLATE_GENERATION_PROMPT.format(
            pyomo_code=self.pyomo_code
        )
        
        # 调用LLM生成模板
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.generate(messages)
        template_code = response.content.strip()
        
        # 提取代码块（如果LLM返回的是markdown格式）
        if "```python" in template_code:
            # 提取```python和```之间的代码
            start = template_code.find("```python") + len("```python")
            end = template_code.find("```", start)
            if end > start:
                template_code = template_code[start:end].strip()
        elif "```" in template_code:
            # 提取```和```之间的代码
            start = template_code.find("```") + len("```")
            end = template_code.find("```", start)
            if end > start:
                template_code = template_code[start:end].strip()
        
        return template_code


def generate_template_from_pyomo(
    llm,
    problem_dir: str
) -> str:
    """
    从Pyomo求解器代码生成LLM求解器模板（便捷函数）
    
    Args:
        llm: 语言模型实例
        problem_dir: 问题目录路径
        
    Returns:
        生成的模板字符串
    """
    generator = TemplateGenerator(llm, problem_dir)
    return generator.generate_template()
