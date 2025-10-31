"""
测试生成的 llm_solver (Task 3)

这个测试脚本模拟智能体中的验证过程，包括：
1. 导入生成的solver
2. 使用evaluation模块的test_llm_solver函数进行完整验证
3. 检查约束是否满足
4. 计算makespan性能

使用方法：
    python test_solver_task3.py
"""
import sys
from pathlib import Path

# 添加项目根目录到path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入evaluation模块
from MMAgent.evaluation import test_llm_solver

# 导入生成的solver
from output.code.solver_task3 import llm_solver


def main():
    """主测试函数"""
    print("="*80)
    print("Testing Generated LLM Solver (Task 3)")
    print("="*80)
    print()
    
    # 测试配置
    test_config = {
        'topo_name': 'FatTree',         # 拓扑类型
        'ina_num_list': [3],            # INA数量列表
        'jobs_num_list': [6],           # 作业数量列表
        'instances_num': 1              # 测试实例数量
    }
    
    print("Test Configuration:")
    print(f"  Topology: {test_config['topo_name']}")
    print(f"  INA counts: {test_config['ina_num_list']}")
    print(f"  Job counts: {test_config['jobs_num_list']}")
    print(f"  Instances per configuration: {test_config['instances_num']}")
    print()
    
    print("="*80)
    print("Running Test...")
    print("="*80)
    print()
    
    try:
        # 调用test_llm_solver进行完整测试
        results = test_llm_solver(
            solver_func=llm_solver,
            topo_name=test_config['topo_name'],
            ina_num_list=test_config['ina_num_list'],
            jobs_num_list=test_config['jobs_num_list'],
            instances_num=test_config['instances_num']
        )
        
        # 打印结果
        print("\n" + "="*80)
        print("Test Results Summary")
        print("="*80)
        
        all_valid = True
        for config_key, jct_list in results.items():
            ina_count, job_count = config_key
            print(f"\nConfiguration: INA={ina_count}, Jobs={job_count}")
            print(f"  Number of instances: {len(jct_list)}")
            
            # 检查是否有无效结果
            valid_jcts = [jct for jct in jct_list if jct < float('inf')]
            invalid_count = len(jct_list) - len(valid_jcts)
            
            if invalid_count > 0:
                all_valid = False
                print(f"  ❌ Invalid results: {invalid_count}/{len(jct_list)}")
            else:
                print(f"  ✓ All results valid")
            
            if valid_jcts:
                avg_jct = sum(valid_jcts) / len(valid_jcts)
                min_jct = min(valid_jcts)
                max_jct = max(valid_jcts)
                
                print(f"  Makespan Statistics:")
                print(f"    Average: {avg_jct:.4f}")
                print(f"    Min:     {min_jct:.4f}")
                print(f"    Max:     {max_jct:.4f}")
            else:
                print(f"  ❌ No valid makespans computed")
        
        print("\n" + "="*80)
        if all_valid:
            print("✓ TEST PASSED - All constraints satisfied")
        else:
            print("✗ TEST FAILED - Some results invalid or constraints violated")
        print("="*80)
        
        return 0 if all_valid else 1
        
    except ValueError as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED - Constraint Violation")
        print("="*80)
        print(f"\nError: {str(e)}")
        print("\nThe generated solver violates LLMINA constraints.")
        print("Please check the error message above for details.")
        print("="*80)
        return 1
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED - Unexpected Error")
        print("="*80)
        print(f"\nError Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("\nPlease check the solver implementation.")
        print("="*80)
        
        import traceback
        print("\nFull Traceback:")
        traceback.print_exc()
        return 1


def run_detailed_test():
    """运行更详细的测试，测试多种配置"""
    print("="*80)
    print("Detailed Testing - Multiple Configurations")
    print("="*80)
    print()
    
    # 多种测试配置
    test_configs = [
        {
            'name': 'Small Scale',
            'topo_name': 'FatTree',
            'ina_num_list': [3],
            'jobs_num_list': [6],
            'instances_num': 3
        },
        {
            'name': 'Medium Scale',
            'topo_name': 'FatTree',
            'ina_num_list': [5],
            'jobs_num_list': [10],
            'instances_num': 2
        },
    ]
    
    all_passed = True
    
    for config in test_configs:
        print("\n" + "="*80)
        print(f"Testing: {config['name']}")
        print("="*80)
        print(f"  Topology: {config['topo_name']}")
        print(f"  INA: {config['ina_num_list']}")
        print(f"  Jobs: {config['jobs_num_list']}")
        print(f"  Instances: {config['instances_num']}")
        print()
        
        try:
            results = test_llm_solver(
                solver_func=llm_solver,
                topo_name=config['topo_name'],
                ina_num_list=config['ina_num_list'],
                jobs_num_list=config['jobs_num_list'],
                instances_num=config['instances_num']
            )
            
            # 检查结果
            config_passed = True
            for config_key, jct_list in results.items():
                valid_jcts = [jct for jct in jct_list if jct < float('inf')]
                if len(valid_jcts) != len(jct_list):
                    config_passed = False
                    all_passed = False
                    print(f"  ❌ Failed: {len(jct_list) - len(valid_jcts)}/{len(jct_list)} invalid")
                else:
                    avg_jct = sum(valid_jcts) / len(valid_jcts)
                    print(f"  ✓ Passed: Avg Makespan = {avg_jct:.4f}")
            
            if config_passed:
                print(f"\n✓ {config['name']}: PASSED")
            else:
                print(f"\n✗ {config['name']}: FAILED")
                
        except Exception as e:
            print(f"\n✗ {config['name']}: ERROR - {str(e)}")
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test generated LLM solver')
    parser.add_argument('--detailed', action='store_true', 
                        help='Run detailed tests with multiple configurations')
    
    args = parser.parse_args()
    
    if args.detailed:
        exit_code = run_detailed_test()
    else:
        exit_code = main()
    
    sys.exit(exit_code)
