"""
简化版 LangGraph 主程序
移除不必要的封装，直接使用 LangGraph 内置功能
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm.llm import LLM
from utils.utils import write_json_file, load_config
from langgraph_core.state import create_agent_state
from langgraph_core.graph import build_workflow
from datetime import datetime
import argparse
from evaluator_interface import load_evaluator

def main():
    """简化的主函数"""
    parser = argparse.ArgumentParser(description='LLM Solver')
    parser.add_argument('--task', type=str, default='LLMINA', help='Task name (use LLMINA for the specialized solver)')
    parser.add_argument('--checkpoint', action='store_true', help='Enable checkpoints')
    args = parser.parse_args()
    
    # 获取配置（从 config.yaml）
    config = load_config()
    
    # 初始化 LLM - 使用 ChatOpenAI
    llm = LLM(
        model_name=config.get('model_name', 'qwen3-30b-a3b'),
        key=config.get('api_key', 'any'),
        api_base=config.get('api_base', 'http://192.168.1.44:8021/v1')
    )
    
    # 构建工作流 - 直接使用简化版本
    workflow = build_workflow(
        enable_checkpoints=args.checkpoint or config.get('enable_checkpoints', False),
        checkpoint_path=config.get('checkpoint_path')
    )
    
    # 构建输出目录：output/{task}_{datetime}
    
    base_output_dir = config.get('paths').get('output')
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = Path(base_output_dir) / f"{args.task}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    problem_dir = config.get('paths').get('problem', 'MMBench/problem')
    task_dir = Path(problem_dir) / args.task
    problem_path = task_dir / "problem.json"
    
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    if not problem_path.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_path}")
    
    # 将 task_dir 添加到配置中，供其他模块使用
    config['task_dir'] = str(task_dir / 'runtime')
    config['template_dir'] = str(task_dir / 'code_template')
    config['problem_dir'] = str(task_dir)  # 问题目录路径
    
    # 动态加载用户的评估模块    
    evaluator = load_evaluator(str(task_dir))
    config['test_solver_func'] = evaluator['test_solver']
    config['helper_functions'] = evaluator['helper_functions']
    config['runtime_dir'] = evaluator.get('runtime_dir', str(task_dir))  # 保存 runtime 目录路径
    print(f"✓ Loaded evaluation module")

    
    # 创建初始状态
    initial_state = create_agent_state(
        config=config,
        llm=llm,
        task_name=args.task,
        task_dir=str(task_dir),
        problem_path=str(problem_path),
        output_dir=str(output_dir)
    )
    
    start = time.time()
    
    # try:
    # 执行工作流 - 简单直接
    final_state = workflow.invoke(initial_state)
    
    # 提取结果
    solver_path = final_state.get('solver_code_path', '')
    errors = final_state.get('errors', [])
    
    print(f"\n{'='*80}")
    print(f"Completed in {time.time() - start:.2f}s")
    print(f"Solver: {solver_path}")
    print(f"Errors: {len(errors)}")
    print(f"{'='*80}\n")
    
    # 保存结果
    write_json_file(f'{output_dir}/usage/{args.task}.json', llm.get_total_usage())
    write_json_file(f'{output_dir}/usage/{args.task}_result.json', {
        'status': 'success' if solver_path else 'failed',
        'solver_path': solver_path,
        'errors': errors,
        'llm_usage': llm.get_total_usage()
    })
    
    with open(f'{output_dir}/usage/runtime.txt', 'w') as f:
        f.write(f"{time.time() - start:.2f}s")
    
    return solver_path        

if __name__ == "__main__":
    main()
