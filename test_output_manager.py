"""
测试输出管理器功能

验证中间结果保存功能是否正常工作
"""
import sys
from pathlib import Path
import tempfile
import shutil

# 添加项目根目录到path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from MMAgent.utils.output_manager import (
    ensure_output_dirs,
    save_modeling_solution,
    save_task_descriptions,
    save_dependency_info,
    save_task_code,
    save_task_result,
    save_complete_solution,
    save_workflow_log,
    create_readme
)


def test_output_manager():
    """测试输出管理器的所有功能"""
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='llmina_test_')
    print(f"Testing in temporary directory: {temp_dir}")
    print()
    
    try:
        # 1. 测试目录创建
        print("1. Testing directory creation...")
        dirs = ensure_output_dirs(temp_dir)
        print(f"   ✓ Created directories: {list(dirs.keys())}")
        print()
        
        # 2. 测试建模方案保存
        print("2. Testing modeling solution save...")
        modeling_solution = "This is a test modeling solution with algorithm design."
        path = save_modeling_solution(temp_dir, modeling_solution)
        print(f"   ✓ Saved to: {path}")
        print()
        
        # 3. 测试任务描述保存
        print("3. Testing task descriptions save...")
        task_descriptions = [
            "Task 1: Initialize data structures",
            "Task 2: Implement core algorithm",
            "Task 3: Optimize performance"
        ]
        path = save_task_descriptions(temp_dir, task_descriptions)
        print(f"   ✓ Saved to: {path}")
        print()
        
        # 4. 测试依赖信息保存
        print("4. Testing dependency info save...")
        dependency_dag = {"1": [], "2": ["1"], "3": ["1", "2"]}
        dependency_analysis = ["No dependencies", "Depends on Task 1", "Depends on Tasks 1 and 2"]
        execution_order = [1, 2, 3]
        paths = save_dependency_info(temp_dir, dependency_dag, dependency_analysis, execution_order)
        print(f"   ✓ Saved DAG to: {paths['dag']}")
        print(f"   ✓ Saved analysis to: {paths['analysis']}")
        print()
        
        # 5. 测试任务代码保存
        print("5. Testing task code save...")
        task_code = """def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    # Task 1 implementation
    return placement, routing
"""
        path = save_task_code(temp_dir, 1, task_code)
        print(f"   ✓ Saved to: {path}")
        print()
        
        # 6. 测试任务结果保存
        print("6. Testing task result save...")
        task_result = {
            'task_code': task_code,
            'is_pass': True,
            'execution_result': 'Test passed with makespan=123.45',
            'validation_result': {'is_valid': True}
        }
        path = save_task_result(temp_dir, 1, task_descriptions[0], task_result)
        print(f"   ✓ Saved to: {path}")
        print()
        
        # 7. 测试工作流日志
        print("7. Testing workflow log...")
        save_workflow_log(temp_dir, 'test_event', {'data': 'test'}, level='INFO')
        print(f"   ✓ Log entry added")
        print()
        
        # 8. 测试完整解决方案保存
        print("8. Testing complete solution save...")
        mock_state = {
            'problem_path': '/path/to/problem.json',
            'modeling_solution': modeling_solution,
            'total_tasks': 3,
            'execution_order': execution_order,
            'task_results': {
                1: {'task_description': task_descriptions[0], 'is_pass': True},
                2: {'task_description': task_descriptions[1], 'is_pass': True},
                3: {'task_description': task_descriptions[2], 'is_pass': False}
            },
            'solver_code_path': f'{temp_dir}/llm_solver_final.py',
            'evaluation_metrics': {'avg_makespan': 123.45}
        }
        path = save_complete_solution(temp_dir, mock_state)
        print(f"   ✓ Saved to: {path}")
        print()
        
        # 9. 测试README创建
        print("9. Testing README creation...")
        path = create_readme(temp_dir, mock_state)
        print(f"   ✓ Created at: {path}")
        print()
        
        # 列出生成的文件
        print("="*80)
        print("Generated files:")
        print("="*80)
        for root, dirs, files in Path(temp_dir).walk():
            level = root.relative_to(temp_dir).parts
            indent = ' ' * 2 * len(level)
            print(f'{indent}{root.name}/')
            subindent = ' ' * 2 * (len(level) + 1)
            for file in files:
                print(f'{subindent}{file}')
        print()
        
        print("="*80)
        print("✓ All tests passed!")
        print("="*80)
        print(f"\nTest output saved in: {temp_dir}")
        print("You can inspect the files to verify the output format.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 可选：清理临时目录
        # shutil.rmtree(temp_dir)
        pass


if __name__ == '__main__':
    success = test_output_manager()
    sys.exit(0 if success else 1)
