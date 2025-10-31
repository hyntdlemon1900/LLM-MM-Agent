"""
通用 LLM Solver 测试脚本

这个测试脚本可以测试任意生成的solver，模拟智能体中的验证过程。

功能：
1. 动态导入指定的solver文件
2. 使用evaluation模块进行完整验证
3. 约束检查（INA预算、路由完整性、容量限制等）
4. 性能评估（makespan计算）

使用方法：
    # 测试特定task的solver
    python test_solver.py --task 3
    
    # 测试指定路径的solver文件
    python test_solver.py --file output/code/solver_task3.py
    
    # 运行详细测试（多种配置）
    python test_solver.py --task 3 --detailed
    
    # 自定义测试参数
    python test_solver.py --task 3 --ina 3 5 --jobs 6 10 --instances 5
"""
import sys
import os
from pathlib import Path
import importlib.util
import argparse

# 添加项目根目录到path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入evaluation模块
from MMAgent.evaluation import test_llm_solver


def load_solver_from_file(solver_path: str):
    """
    从文件动态导入solver函数
    
    Args:
        solver_path: solver文件路径
        
    Returns:
        llm_solver函数
    """
    solver_path = Path(solver_path)
    
    if not solver_path.exists():
        raise FileNotFoundError(f"Solver file not found: {solver_path}")
    
    # 动态导入模块
    module_name = solver_path.stem
    spec = importlib.util.spec_from_file_location(module_name, solver_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # 获取llm_solver函数
    if not hasattr(module, 'llm_solver'):
        raise AttributeError(f"Module {solver_path} does not contain 'llm_solver' function")
    
    return module.llm_solver


def run_test(solver_func, config: dict, verbose: bool = True):
    """
    运行solver测试
    
    Args:
        solver_func: solver函数
        config: 测试配置字典
        verbose: 是否打印详细信息
        
    Returns:
        (success, results): 测试是否成功和结果字典
    """
    if verbose:
        print("="*80)
        print("Test Configuration:")
        print(f"  Topology: {config['topo_name']}")
        print(f"  INA counts: {config['ina_num_list']}")
        print(f"  Job counts: {config['jobs_num_list']}")
        print(f"  Instances per configuration: {config['instances_num']}")
        print("="*80)
        print()
    
    try:
        # 调用test_llm_solver进行完整测试
        results = test_llm_solver(
            solver_func=solver_func,
            topo_name=config['topo_name'],
            ina_num_list=config['ina_num_list'],
            jobs_num_list=config['jobs_num_list'],
            instances_num=config['instances_num']
        )
        
        # 分析结果
        all_valid = True
        summary = {}
        
        for config_key, jct_list in results.items():
            ina_count, job_count = config_key
            
            # 检查有效性
            valid_jcts = [jct for jct in jct_list if jct < float('inf')]
            invalid_count = len(jct_list) - len(valid_jcts)
            
            if invalid_count > 0:
                all_valid = False
            
            # 统计信息
            if valid_jcts:
                summary[config_key] = {
                    'valid': len(valid_jcts),
                    'invalid': invalid_count,
                    'avg_makespan': sum(valid_jcts) / len(valid_jcts),
                    'min_makespan': min(valid_jcts),
                    'max_makespan': max(valid_jcts),
                    'all_valid': invalid_count == 0
                }
            else:
                summary[config_key] = {
                    'valid': 0,
                    'invalid': invalid_count,
                    'avg_makespan': float('inf'),
                    'min_makespan': float('inf'),
                    'max_makespan': float('inf'),
                    'all_valid': False
                }
        
        if verbose:
            print_results(summary)
        
        return all_valid, summary
        
    except ValueError as e:
        if verbose:
            print("\n" + "="*80)
            print("❌ TEST FAILED - Constraint Violation")
            print("="*80)
            print(f"\nError: {str(e)}")
            print("="*80)
        return False, {'error': str(e), 'type': 'constraint_violation'}
        
    except Exception as e:
        if verbose:
            print("\n" + "="*80)
            print("❌ TEST FAILED - Unexpected Error")
            print("="*80)
            print(f"\nError Type: {type(e).__name__}")
            print(f"Error Message: {str(e)}")
            print("="*80)
            
            import traceback
            print("\nFull Traceback:")
            traceback.print_exc()
        
        return False, {'error': str(e), 'type': type(e).__name__}


def print_results(summary: dict):
    """打印测试结果摘要"""
    print("\n" + "="*80)
    print("Test Results Summary")
    print("="*80)
    
    for config_key, stats in summary.items():
        if 'error' in stats:
            print(f"\n❌ Error: {stats['type']}")
            print(f"   Message: {stats['error']}")
            continue
        
        ina_count, job_count = config_key
        print(f"\nConfiguration: INA={ina_count}, Jobs={job_count}")
        print(f"  Valid instances:   {stats['valid']}")
        print(f"  Invalid instances: {stats['invalid']}")
        
        if stats['all_valid']:
            print(f"  ✓ Status: ALL VALID")
        else:
            print(f"  ✗ Status: SOME INVALID")
        
        if stats['valid'] > 0:
            print(f"  Makespan Statistics:")
            print(f"    Average: {stats['avg_makespan']:.4f}")
            print(f"    Min:     {stats['min_makespan']:.4f}")
            print(f"    Max:     {stats['max_makespan']:.4f}")
    
    print("\n" + "="*80)
    all_passed = all(stats.get('all_valid', False) for stats in summary.values() if 'error' not in stats)
    if all_passed:
        print("✓ TEST PASSED - All constraints satisfied")
    else:
        print("✗ TEST FAILED - Some results invalid or constraints violated")
    print("="*80)


def run_detailed_test(solver_func, verbose: bool = True):
    """运行多种配置的详细测试"""
    if verbose:
        print("="*80)
        print("Detailed Testing - Multiple Configurations")
        print("="*80)
        print()
    
    # 多种测试配置
    test_configs = [
        {
            'name': 'Small Scale (Quick)',
            'topo_name': 'FatTree',
            'ina_num_list': [3],
            'jobs_num_list': [6],
            'instances_num': 2
        },
        {
            'name': 'Medium Scale',
            'topo_name': 'FatTree',
            'ina_num_list': [3, 5],
            'jobs_num_list': [6, 10],
            'instances_num': 1
        },
    ]
    
    all_passed = True
    all_results = {}
    
    for config in test_configs:
        if verbose:
            print("\n" + "="*80)
            print(f"Testing: {config['name']}")
            print("="*80)
        
        success, results = run_test(solver_func, config, verbose=False)
        all_results[config['name']] = (success, results)
        
        if verbose:
            # 打印简要结果
            if success:
                print(f"✓ {config['name']}: PASSED")
                for key, stats in results.items():
                    if 'avg_makespan' in stats and stats['avg_makespan'] < float('inf'):
                        print(f"  INA={key[0]}, Jobs={key[1]}: Avg={stats['avg_makespan']:.4f}")
            else:
                print(f"✗ {config['name']}: FAILED")
                if 'error' in results:
                    print(f"  Error: {results['error']}")
        
        if not success:
            all_passed = False
    
    if verbose:
        print("\n" + "="*80)
        if all_passed:
            print("✓ ALL DETAILED TESTS PASSED")
        else:
            print("✗ SOME DETAILED TESTS FAILED")
        print("="*80)
    
    return all_passed, all_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Test LLM-generated solver functions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test solver from Task 3
  python test_solver.py --task 3
  
  # Test specific solver file
  python test_solver.py --file output/code/my_solver.py
  
  # Run detailed tests
  python test_solver.py --task 3 --detailed
  
  # Custom test configuration
  python test_solver.py --task 3 --ina 3 5 --jobs 6 10 --instances 5 --topo FatTree
        """
    )
    
    # Solver source
    solver_group = parser.add_mutually_exclusive_group(required=True)
    solver_group.add_argument('--task', type=int,
                              help='Task number (will load output/code/solver_task{N}.py)')
    solver_group.add_argument('--file', type=str,
                              help='Path to solver file')
    
    # Test options
    parser.add_argument('--detailed', action='store_true',
                        help='Run detailed tests with multiple configurations')
    parser.add_argument('--topo', type=str, default='FatTree',
                        help='Topology name (default: FatTree)')
    parser.add_argument('--ina', nargs='+', type=int, default=[3],
                        help='INA counts to test (default: 3)')
    parser.add_argument('--jobs', nargs='+', type=int, default=[6],
                        help='Job counts to test (default: 6)')
    parser.add_argument('--instances', type=int, default=1,
                        help='Number of instances per configuration (default: 1)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # 确定solver文件路径
    if args.task is not None:
        solver_path = project_root / 'output' / 'code' / f'solver_task{args.task}.py'
        print(f"Loading solver from Task {args.task}: {solver_path}")
    else:
        solver_path = Path(args.file)
        print(f"Loading solver from: {solver_path}")
    
    print()
    
    # 加载solver
    try:
        solver_func = load_solver_from_file(solver_path)
        print(f"✓ Solver loaded successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to load solver: {e}")
        return 1
    
    # 运行测试
    if args.detailed:
        success, _ = run_detailed_test(solver_func, verbose=not args.quiet)
    else:
        config = {
            'topo_name': args.topo,
            'ina_num_list': args.ina,
            'jobs_num_list': args.jobs,
            'instances_num': args.instances
        }
        success, _ = run_test(solver_func, config, verbose=not args.quiet)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
