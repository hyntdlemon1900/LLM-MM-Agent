"""
配置管理模块
统一管理所有 LangGraph 相关配置
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import json
import os


@dataclass
class ModelConfig:
    """LLM 模型配置"""
    model_name: str = "qwen3-30b-a3b"
    api_key: str = "any"
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    

@dataclass
class WorkflowConfig:
    """工作流配置"""
    max_loops: int = 10
    enable_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    enable_monitoring: bool = True
    enable_visualization: bool = False
    timeout_seconds: Optional[int] = None
    

@dataclass
class NodeConfig:
    """节点配置"""
    # 问题澄清配置
    clarification_max_rounds: int = 3
    clarification_enabled: bool = True
    
    # 问题建模配置
    modeling_max_rounds: int = 2
    modeling_enabled: bool = True
    
    # 任务分解配置
    task_decompose_enabled: bool = True
    default_task_num: int = 3
    min_task_num: int = 2
    max_task_num: int = 5
    
    # 任务求解配置
    task_solving_max_retries: int = 3
    enable_code_validation: bool = True
    enable_progressive_solving: bool = True
    
    # 代码验证配置
    validation_timeout: int = 300
    max_validation_attempts: int = 2
    

@dataclass
class ToolsConfig:
    """工具配置"""
    enable_ina_candidates: bool = True
    enable_evaluation: bool = True
    enable_network_analysis: bool = True
    cache_tool_results: bool = True
    tool_timeout: int = 60
    

@dataclass
class OutputConfig:
    """输出配置"""
    save_intermediate_results: bool = True
    save_code_snapshots: bool = True
    save_validation_logs: bool = True
    generate_visualization: bool = False
    output_format: str = "json"  # "json", "markdown", "latex"
    

@dataclass
class LangGraphConfig:
    """完整的 LangGraph 配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    nodes: NodeConfig = field(default_factory=NodeConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # 自定义配置
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LangGraphConfig':
        """从字典创建配置"""
        config = cls()
        
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'workflow' in data:
            config.workflow = WorkflowConfig(**data['workflow'])
        if 'nodes' in data:
            config.nodes = NodeConfig(**data['nodes'])
        if 'tools' in data:
            config.tools = ToolsConfig(**data['tools'])
        if 'output' in data:
            config.output = OutputConfig(**data['output'])
        if 'custom' in data:
            config.custom = data['custom']
            
        return config
    
    @classmethod
    def from_json(cls, json_path: str) -> 'LangGraphConfig':
        """从 JSON 文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_json(self, json_path: str):
        """保存配置到 JSON 文件"""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def merge(self, other: Dict[str, Any]) -> 'LangGraphConfig':
        """合并外部配置"""
        config_dict = self.to_dict()
        
        def deep_merge(base: Dict, override: Dict) -> Dict:
            """深度合并字典"""
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged = deep_merge(config_dict, other)
        return LangGraphConfig.from_dict(merged)
    

# ============ 预定义配置 ============

def get_llmina_config() -> LangGraphConfig:
    """获取 LLMINA 专用配置"""
    config = LangGraphConfig()
    config.nodes.clarification_enabled = True
    config.nodes.clarification_max_rounds = 3
    config.nodes.modeling_max_rounds = 2
    config.nodes.default_task_num = 3
    config.nodes.enable_progressive_solving = True
    config.workflow.max_loops = 15
    config.output.save_intermediate_results = True
    return config


def get_standard_modeling_config() -> LangGraphConfig:
    """获取标准建模配置"""
    config = LangGraphConfig()
    config.nodes.clarification_enabled = False
    config.nodes.modeling_max_rounds = 3
    config.nodes.default_task_num = 5
    config.workflow.max_loops = 20
    config.output.generate_visualization = True
    return config


def get_quick_test_config() -> LangGraphConfig:
    """获取快速测试配置"""
    config = LangGraphConfig()
    config.nodes.clarification_max_rounds = 1
    config.nodes.modeling_max_rounds = 1
    config.nodes.default_task_num = 2
    config.nodes.max_validation_attempts = 1
    config.workflow.max_loops = 5
    config.workflow.enable_checkpoints = False
    config.output.save_intermediate_results = False
    return config


def get_production_config() -> LangGraphConfig:
    """获取生产环境配置"""
    config = LangGraphConfig()
    config.workflow.enable_checkpoints = True
    config.workflow.enable_monitoring = True
    config.workflow.timeout_seconds = 3600
    config.nodes.task_solving_max_retries = 5
    config.output.save_intermediate_results = True
    config.output.save_validation_logs = True
    return config


# ============ 配置加载函数 ============

def load_config(
    config_path: Optional[str] = None,
    preset: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> LangGraphConfig:
    """
    加载配置
    
    Args:
        config_path: 配置文件路径
        preset: 预设配置名称 ('llmina', 'standard', 'quick_test', 'production')
        overrides: 覆盖配置
        
    Returns:
        LangGraphConfig 实例
    """
    # 1. 从预设或文件加载基础配置
    if config_path:
        config = LangGraphConfig.from_json(config_path)
    elif preset:
        preset_map = {
            'llmina': get_llmina_config,
            'standard': get_standard_modeling_config,
            'quick_test': get_quick_test_config,
            'production': get_production_config
        }
        if preset not in preset_map:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(preset_map.keys())}")
        config = preset_map[preset]()
    else:
        config = LangGraphConfig()
    
    # 2. 应用覆盖配置
    if overrides:
        config = config.merge(overrides)
    
    return config


# ============ 环境变量支持 ============

def load_config_from_env() -> Dict[str, Any]:
    """从环境变量加载配置"""
    env_config = {}
    
    # Model config
    if api_key := os.getenv('LLM_API_KEY'):
        env_config.setdefault('model', {})['api_key'] = api_key
    if model_name := os.getenv('LLM_MODEL_NAME'):
        env_config.setdefault('model', {})['model_name'] = model_name
    if api_base := os.getenv('LLM_API_BASE'):
        env_config.setdefault('model', {})['api_base'] = api_base
    
    # Workflow config
    if max_loops := os.getenv('WORKFLOW_MAX_LOOPS'):
        env_config.setdefault('workflow', {})['max_loops'] = int(max_loops)
    if checkpoint_dir := os.getenv('CHECKPOINT_DIR'):
        env_config.setdefault('workflow', {})['checkpoint_dir'] = checkpoint_dir
    
    return env_config
