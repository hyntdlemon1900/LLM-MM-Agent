"""
简化版 LangGraph 主程序
移除不必要的封装，直接使用 LangGraph 内置功能
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm.llm import LLM
from utils.utils import write_json_file, get_info
from langgraph_core.state import create_llmina_state
from langgraph_core.graph_simplified import build_simple_llmina_workflow


def main():
    """简化的主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified LLMINA Solver')
    parser.add_argument('--task', type=str, default='LLMINA', help='Task name (use LLMINA for the specialized solver)')
    parser.add_argument('--checkpoint', action='store_true', help='Enable checkpoints')
    parser.add_argument('--viz', action='store_true', help='Visualize workflow')
    args = parser.parse_args()
    
    # 获取配置（从 config.yaml）
    problem_path, config, dataset_dir, output_dir = get_info(args)
    
    # 初始化 LLM
    llm = LLM(config['model_name'], config.get('api_key'))
    
    # 构建工作流 - 直接使用简化版本
    workflow = build_simple_llmina_workflow(
        enable_checkpoints=args.checkpoint or config.get('enable_checkpoints', False),
        checkpoint_path=config.get('checkpoint_path')
    )
    
    # 可视化（如果需要）
    if args.viz or config.get('visualize', False):
        from langgraph_core.graph_simplified import visualize_workflow
        visualize_workflow(workflow, config.get('viz_output', 'workflow.png'))
    
    # 创建初始状态
    initial_state = create_llmina_state(
        problem_path=problem_path,
        config=config,
        output_dir=output_dir,
        name=args.task,
        llm=llm
    )
    
    start = time.time()
    
    try:
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
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        
        write_json_file(f'{output_dir}/usage/error.json', {
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        
        raise


if __name__ == "__main__":
    main()
