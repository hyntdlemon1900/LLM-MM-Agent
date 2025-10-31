"""
LangGraph 主程序入口
基于完整 LangGraph 架构的多智能体系统
所有配置从 config.yaml 读取
"""

import sys
import os
import time
from pathlib import Path
import argparse
# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from llm.llm import LLM
from utils.utils import write_json_file, get_info
from langgraph_core.config import get_quick_test_config
from langgraph_core.state import create_llmina_state
from langgraph_core.workflows.llmina_workflow import build_llmina_workflow
from evaluation import FixedINAEvaluation


def run_llmina(problem_path: str, config: dict, output_dir: str, name: str):
    """
    运行 LLMINA 求解流程（LangGraph 版本）
    
    Args:
        problem_path: 问题路径
        config: 配置字典
        output_dir: 输出目录
        name: 任务名称
        
    Returns:
        (solver_code_path, result_dict)
    """
    # 初始化 LLM
    api_key = config.get('api_key')
    llm = LLM(config['model_name'], api_key)
    
    # 从 config 字典读取配置（已通过 get_info → load_config 加载 config.yaml）
    langgraph_config = get_quick_test_config()
    langgraph_config.nodes.clarification_max_rounds = config.get('clarification_rounds', 1)
    langgraph_config.nodes.modeling_max_rounds = config.get('modeling_rounds', 1)
    langgraph_config.nodes.default_task_num = config.get('default_task_num', 2)
    config['max_loops'] = 10
    
    # 可视化工作流（如果需要）
    if config.get('visualize', False):
        from langgraph_core.workflows.llmina_workflow import visualize_llmina_workflow
        viz_output = config.get('viz_output', 'workflow.png')
        visualize_llmina_workflow(viz_output, langgraph_config)
    
    # 构建工作流
    workflow = build_llmina_workflow(langgraph_config)
    
    # 创建初始状态
    initial_state = create_llmina_state(
        problem_path=problem_path,
        config=config,
        output_dir=output_dir,
        name=name,
        llm=llm
    )
    
    print(f"[LLMINA Workflow] Starting execution...")

    # 执行工作流
    try:
        final_state = workflow.invoke(initial_state)
        
        print(f"[LLMINA Workflow] Execution completed")
        
        # 提取结果
        solver_code_path = final_state.get('solver_code_path', '')
        status = final_state.get('status', 'unknown')
        errors = final_state.get('errors', [])
        task_results = final_state.get('task_results', {})
        
        print(f"Status: {status}")
        print(f"Solver code path: {solver_code_path}")
        print(f"Task results: {len(task_results)} tasks")
        
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for i, error in enumerate(errors[:5], 1):  # 只显示前5个错误
                print(f"  {i}. {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors)-5} more errors")
        
        result = {
            'status': 'success' if solver_code_path else 'partial',
            'solver_code_path': solver_code_path,
            'task_results': task_results,
            'errors': errors
        }
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"[LLMINA Workflow] Execution failed: {e}")
        print(f"{'='*80}\n")
        
        import traceback
        traceback.print_exc()
        
        # 保存错误信息
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        write_json_file(f'{output_dir}/usage/error.json', error_info)
        
        result = {
            'status': 'failed',
            'error': str(e),
            'errors': [str(e)],
            'traceback': traceback.format_exc()
        }
        
        raise
    
    # 保存统计信息
    print('\nLLM Usage:', llm.get_total_usage())
    write_json_file(f'{output_dir}/usage/{name}.json', llm.get_total_usage())
    
    # 保存结果
    write_json_file(f'{output_dir}/usage/{name}_result.json', {
        'status': result['status'],
        'solver_path': result.get('solver_code_path'),
        'errors': result.get('errors', []),
        'llm_usage': llm.get_total_usage()
    })
    
    return result.get('solver_code_path'), result


def evaluate_llmina_solver(solver_code_path, topo_name='FatTree', ina_num_list=[3],
                           jobs_num_list=[10], instances_num=10):
    """
    使用生成的 solver 代码进行评估
    
    Args:
        solver_code_path: Solver 代码路径
        topo_name: 拓扑名称
        ina_num_list: INA 数量列表
        jobs_num_list: 任务数量列表
        instances_num: 实例数量
        
    Returns:
        评估结果
    """
    if not solver_code_path or not os.path.exists(solver_code_path):
        print(f"[Evaluation] Solver code not found: {solver_code_path}")
        return None
    
    print('\n' + '=' * 80)
    print('Evaluating LLMINA Solver')
    print('=' * 80)
    
    # 动态导入生成的 solver 模块
    import importlib.util
    spec = importlib.util.spec_from_file_location("llm_solver", solver_code_path)
    llm_solver_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llm_solver_module)
    
    # 创建评估类的子类
    class LLMINAEvaluationWithLLMSolver(FixedINAEvaluation):
        def solver(self, instance):
            """使用 LLM 生成的 solver 替换原有实现"""
            return llm_solver_module.llm_solver(
                instance=instance,
                network=self.network,
                ina_num=self.ina_num,
                jobs_num=self.jobs_num,
                Cs=self.Cs,
                topo_name=self.topo_name
            )
    
    # 运行评估
    evaluator = LLMINAEvaluationWithLLMSolver(
        topo_name=topo_name,
        ina_num_list=ina_num_list,
        jobs_num_list=jobs_num_list,
        instances_num=instances_num
    )
    
    results = evaluator.evaluate_program()
    
    print('\n' + '=' * 80)
    print('Evaluation Results:')
    print(results)
    print('=' * 80)
    
    return results


def main():
    """主函数 - 所有配置从 config.yaml 读取"""
    parser = argparse.ArgumentParser(description='LLMINA Specialized Solver')
    parser.add_argument('--task', type=str, default='llmina', help='Task name (must be LLMINA)')    
    args = parser.parse_args()
    
    # 获取配置信息
    problem_path, config, dataset_dir, output_dir = get_info(args)
    
    # 记录开始时间
    start = time.time()
    
    # 运行 LLMINA 求解流程
    solver_code_path, solution_dict = run_llmina(
        problem_path=problem_path,
        config=config,
        output_dir=output_dir,
        name=args.task
    )
    
    # 记录结束时间
    end = time.time()
    runtime = end - start
    
    with open(output_dir + '/usage/runtime.txt', 'w') as f:
        f.write(f"{runtime:.2f}s")
    
    print(f"\nTotal runtime: {runtime:.2f}s")
    
    # 如果需要，运行评估（从配置读取）
    eval_config = config.get('evaluation', {})
    if eval_config.get('enable', False) and solver_code_path:
        eval_results = evaluate_llmina_solver(
            solver_code_path=solver_code_path,
            topo_name=eval_config.get('topo_name', 'FatTree'),
            ina_num_list=eval_config.get('ina_num_list', [3]),
            jobs_num_list=eval_config.get('jobs_num_list', [10]),
            instances_num=eval_config.get('instances_num', 10)
        )
        
        if eval_results:
            # 保存评估结果
            write_json_file(f'{output_dir}/usage/evaluation_results.json', eval_results)
    
    print(f"\n{'='*80}")
    print("LangGraph Multi-Agent System Completed")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
