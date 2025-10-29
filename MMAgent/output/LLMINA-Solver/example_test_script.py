"""
示例测试脚本
展示如何使用evaluation.py框架验证LLM生成的solver
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "MMAgent"))

# 导入评估框架
from MMAgent.evaluation import test_llm_solver

# 导入生成的solver（假设已经生成了task_1.py）
# 在实际使用中，这会动态导入生成的solver模块
# solver_module = __import__('task_1')
# llm_solver_func = getattr(solver_module, 'llm_solver')

def example_llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    """
    示例solver函数
    在实际使用中，这将是LLM生成的solver
    """
    from MMAgent.code_template.llmina_solver_template import (
        get_ina_candidates, evaluate_completion_time
    )
    
    # 获取候选INA交换机
    ina_candidates = get_ina_candidates(network)
    max_workers = max(instance['workers_num'])
    num_candidates = len(ina_candidates)
    
    # 简单策略：选择前K个候选
    ina_placement_expanded = [1 if i < K else 0 for i in range(num_candidates)]
    
    # 简单路由：每个worker直接连接到PS
    y_j_w_s_full = [[[0] * num_candidates for _ in range(max_workers)] for _ in range(jobs_num)]
    y_j_w_d = [[1 if w < instance['workers_num'][j] else 0 for w in range(max_workers)] 
               for j in range(jobs_num)]
    jobs_routing_expanded = [y_j_w_s_full, y_j_w_d]
    
    # 评估性能
    makespan = evaluate_completion_time(
        instance, network, ina_placement_expanded, jobs_routing_expanded, verbose=False
    )
    
    return ina_placement_expanded, jobs_routing_expanded, makespan


if __name__ == "__main__":
    print("="*80)
    print("Example: Testing LLM-generated solver with evaluation framework")
    print("="*80)
    
    try:
        # 使用test_llm_solver框架测试
        results = test_llm_solver(
            solver_func=example_llm_solver,
            topo_name='FatTree',
            ina_num_list=[3],
            jobs_num_list=[10],
            instances_num=2  # 使用少量实例快速验证
        )
        
        print("\n" + "="*80)
        print("Test Results:")
        for key, jct_list in results.items():
            ina_num, jobs_num = key
            avg_jct = sum(jct_list) / len(jct_list) if jct_list else float('inf')
            print(f"  INA={ina_num}, Jobs={jobs_num}: Avg JCT={avg_jct:.4f}")
            print(f"    Individual JCTs: {jct_list}")
        print("="*80)
        
        # 验证结果有效性
        all_valid = all(
            all(jct < float('inf') for jct in jct_list) 
            for jct_list in results.values()
        )
        
        if all_valid:
            print("\n✓ Solver validation PASSED")
            print("  All instances completed successfully with finite JCT")
        else:
            print("\n✗ Solver validation FAILED")
            print("  Some instances returned invalid results (inf JCT)")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Solver validation FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
