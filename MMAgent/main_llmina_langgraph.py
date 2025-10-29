"""
LLMINA专用求解器主程序
针对LLMINA问题的专门求解流程，包括问题分析、问题建模、算法设计和测试验证
"""
from llm.llm import LLM
from utils.utils import write_json_file, get_info
import time
import argparse
from utils.pipeline_langgraph import llmina_solver_pipeline_langgraph
from evaluation import FixedINAEvaluation


def run_llmina(key, problem_path, config, name, output_dir):
    """
    运行LLMINA专用求解流程 (LangGraph版本)
    """
    # Initialize LLM
    llm = LLM(config['model_name'], key)

    print('=' * 80)
    print('LLMINA Specialized Solver Pipeline (LangGraph)')
    print('=' * 80)

    # 执行LLMINA求解流程，生成solver结果
    solution_dict = llmina_solver_pipeline_langgraph(
        llm=llm,
        problem_path=problem_path,
        config=config,
        output_dir=output_dir,
        name=name
    )

    print('\n' + '=' * 80)
    print('LangGraph Pipeline Result:')
    print(solution_dict)
    print('=' * 80)

    # 保存统计信息
    print('\nUsage:', llm.get_total_usage())
    write_json_file(f'{output_dir}/usage/{name}.json', llm.get_total_usage())

    # pipeline_langgraph目前不生成solver_code_path，返回None
    return None, solution_dict


def evaluate_llmina_solver(solver_code_path, topo_name='FatTree', ina_num_list=[3], 
                           jobs_num_list=[10], instances_num=10):
    """
    使用生成的solver代码进行评估
    """
    print('\n' + '=' * 80)
    print('Evaluating LLMINA Solver')
    print('=' * 80)

    # 动态导入生成的solver模块
    import importlib.util
    spec = importlib.util.spec_from_file_location("llm_solver", solver_code_path)
    llm_solver_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llm_solver_module)

    # 创建评估类的子类，使用LLM生成的solver
    class LLMINAEvaluationWithLLMSolver(FixedINAEvaluation):
        def solver(self, instance):
            """使用LLM生成的solver替换原有实现"""
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


def parse_arguments():
    parser = argparse.ArgumentParser(description='LLMINA Specialized Solver')
    parser.add_argument('--model_name', type=str, default='qwen3-30b-a3b',
                        help='LLM model name')
    parser.add_argument('--method_name', type=str, default='LLMINA-Solver',
                        help='Method name for output directory')
    parser.add_argument('--task', type=str, default='LLMINA',
                        help='Task name (must be LLMINA)')
    parser.add_argument('--key', type=str, default='any',
                        help='API key for LLM')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    
    # Evaluation parameters
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation after generating solver')
    parser.add_argument('--topo_name', type=str, default='FatTree',
                        choices=['FatTree', 'SpineLeaf'],
                        help='Network topology name')
    parser.add_argument('--ina_num', type=int, nargs='+', default=[3],
                        help='List of INA numbers to evaluate')
    parser.add_argument('--jobs_num', type=int, nargs='+', default=[10],
                        help='List of job numbers to evaluate')
    parser.add_argument('--instances_num', type=int, default=10,
                        help='Number of instances for evaluation')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # 验证任务名称
    if args.task != 'LLMINA':
        print("Warning: This script is specialized for LLMINA task. Setting task to 'LLMINA'.")
        args.task = 'LLMINA'

    # 获取配置信息
    problem_path, config, dataset_dir, output_dir = get_info(args)
    
    # 记录开始时间
    start = time.time()

    # 运行LLMINA求解流程
    solver_code_path, solution_dict = run_llmina(
        key=args.key,
        problem_path=problem_path,
        config=config,
        name=args.task,
        output_dir=output_dir
    )

    # 记录结束时间
    end = time.time()
    with open(output_dir + '/usage/runtime.txt', 'w') as f:
        f.write("{:.2f}s".format(end - start))

    print(f"\nTotal runtime: {end - start:.2f}s")

    # 如果需要，运行评估
    if args.evaluate:
        eval_results = evaluate_llmina_solver(
            solver_code_path=solver_code_path,
            topo_name=args.topo_name,
            ina_num_list=args.ina_num,
            jobs_num_list=args.jobs_num,
            instances_num=args.instances_num
        )
        
        # 保存评估结果
        write_json_file(f'{output_dir}/usage/evaluation_results.json', eval_results)
