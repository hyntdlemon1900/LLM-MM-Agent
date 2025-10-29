"""
任务分类智能体：判断子任务类型，决定处理策略
- 算法设计任务：需要知识库检索和复杂建模
- 简单实现任务：直接编码实现
- 测试验证任务：生成测试代码
- 数据处理任务：数据转换或预处理
"""
from .base_agent import BaseAgent
from typing import Dict, List
import json


class TaskClassifier(BaseAgent):
    """任务分类智能体"""
    
    def __init__(self, llm):
        super().__init__(llm)
        self.task_categories = {
            'algorithm_design': '需要设计复杂算法，应检索知识库寻找相似方法',
            'simple_implementation': '简单的实现任务，可直接编码',
            'testing_validation': '测试或验证任务，生成测试代码',
            'data_processing': '数据处理或转换任务',
            'interface_integration': '接口整合或函数调用任务'
        }
    
    def classify_task(self, task_description: str, modeling_solution: str) -> Dict:
        """
        分类子任务类型
        
        Args:
            task_description: 任务描述
            modeling_solution: 高层次建模方案
            
        Returns:
            {
                'category': str,  # 任务类别
                'requires_kb_retrieval': bool,  # 是否需要知识库检索
                'complexity': str,  # 'low', 'medium', 'high'
                'suggested_approach': str,  # 建议的处理方法
                'reasoning': str  # 分类理由
            }
        """
        prompt = self._build_classification_prompt(task_description, modeling_solution)
        response = self.llm.generate(prompt)
        
        # 解析LLM响应
        try:
            result = self._parse_classification_response(response)
            return result
        except Exception as e:
            print(f"[Warning] Task classification parsing failed: {e}")
            # 默认分类：中等复杂度，需要知识库检索
            return {
                'category': 'algorithm_design',
                'requires_kb_retrieval': True,
                'complexity': 'medium',
                'suggested_approach': 'Use standard algorithm design with KB retrieval',
                'reasoning': 'Default classification due to parsing error'
            }
    
    def _build_classification_prompt(self, task_description: str, modeling_solution: str) -> str:
        """构建分类提示"""
        categories_desc = "\n".join([f"- {k}: {v}" for k, v in self.task_categories.items()])
        
        prompt = f"""You are an expert task classifier for mathematical modeling and algorithm design problems.

**Task Categories:**
{categories_desc}

**High-Level Solution:**
{modeling_solution}

**Task to Classify:**
{task_description}

**Instructions:**
Analyze the task and classify it into one of the categories above. Consider:
1. Does this task require designing a complex algorithm or heuristic? (algorithm_design)
2. Is this a straightforward implementation task? (simple_implementation)
3. Is this for testing or validation purposes? (testing_validation)
4. Is this mainly about data transformation? (data_processing)
5. Is this about integrating existing functions or APIs? (interface_integration)

**Output Format (JSON):**
```json
{{
    "category": "<one of: algorithm_design, simple_implementation, testing_validation, data_processing, interface_integration>",
    "requires_kb_retrieval": <true/false>,
    "complexity": "<low/medium/high>",
    "suggested_approach": "<brief description of how to approach this task>",
    "reasoning": "<explanation of why you classified it this way>"
}}
```

Provide your analysis:"""
        return prompt
    
    def _parse_classification_response(self, response: str) -> Dict:
        """解析LLM的分类响应"""
        # 提取JSON部分
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        
        result = json.loads(json_str)
        
        # 验证必需字段
        required_fields = ['category', 'requires_kb_retrieval', 'complexity', 'suggested_approach', 'reasoning']
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        return result
    
    def batch_classify_tasks(self, task_descriptions: List[str], modeling_solution: str) -> List[Dict]:
        """
        批量分类任务
        
        Args:
            task_descriptions: 任务描述列表
            modeling_solution: 高层次建模方案
            
        Returns:
            分类结果列表
        """
        classifications = []
        for i, task_desc in enumerate(task_descriptions):
            print(f"\n[Task Classifier] Classifying task {i+1}/{len(task_descriptions)}...")
            classification = self.classify_task(task_desc, modeling_solution)
            classifications.append(classification)
            print(f"  Category: {classification['category']}")
            print(f"  KB Retrieval: {classification['requires_kb_retrieval']}")
            print(f"  Complexity: {classification['complexity']}")
        
        return classifications
    
    def get_processing_strategy(self, classification: Dict) -> Dict:
        """
        根据分类结果获取处理策略
        
        Returns:
            {
                'use_kb_retrieval': bool,
                'modeling_rounds': int,
                'code_generation_strategy': str,
                'testing_required': bool
            }
        """
        category = classification['category']
        complexity = classification['complexity']
        
        # 根据任务类型和复杂度定制处理策略
        if category == 'algorithm_design':
            return {
                'use_kb_retrieval': True,
                'modeling_rounds': 2 if complexity == 'high' else 1,
                'code_generation_strategy': 'iterative_refinement',
                'testing_required': True,
                'max_debug_iterations': 5
            }
        elif category == 'simple_implementation':
            return {
                'use_kb_retrieval': False,
                'modeling_rounds': 0,
                'code_generation_strategy': 'direct_generation',
                'testing_required': True,
                'max_debug_iterations': 3
            }
        elif category == 'testing_validation':
            return {
                'use_kb_retrieval': False,
                'modeling_rounds': 0,
                'code_generation_strategy': 'test_generation',
                'testing_required': False,
                'max_debug_iterations': 2
            }
        elif category == 'data_processing':
            return {
                'use_kb_retrieval': False,
                'modeling_rounds': 0,
                'code_generation_strategy': 'data_pipeline',
                'testing_required': True,
                'max_debug_iterations': 3
            }
        elif category == 'interface_integration':
            return {
                'use_kb_retrieval': False,
                'modeling_rounds': 0,
                'code_generation_strategy': 'api_integration',
                'testing_required': True,
                'max_debug_iterations': 3
            }
        else:
            # 默认策略
            return {
                'use_kb_retrieval': classification['requires_kb_retrieval'],
                'modeling_rounds': 1,
                'code_generation_strategy': 'standard',
                'testing_required': True,
                'max_debug_iterations': 4
            }
