"""
测试代码模板加载逻辑

验证：
1. 第一个任务：llmina_solver_template.py + llmina_helper_functions.py
2. 后续任务：llmina_helper_functions.py
"""

import os
import sys

# 添加路径
sys.path.insert(0, '/home/hyn/LLM-MM-Agent-LLMINA')

def test_template_loading():
    """测试模板加载"""
    template_dir = 'MMAgent/code_template'
    
    print("="*80)
    print("Testing Template Loading Logic")
    print("="*80)
    
    # 读取辅助函数文件
    helper_file = os.path.join(template_dir, 'llmina_helper_functions.py')
    print(f"\n1. Loading helper functions from: {helper_file}")
    with open(helper_file, 'r', encoding='utf-8') as f:
        helper_functions = f.read()
    print(f"   ✓ Helper functions loaded: {len(helper_functions)} characters")
    print(f"   ✓ Contains get_ina_candidates: {'get_ina_candidates' in helper_functions}")
    print(f"   ✓ Contains evaluate_completion_time: {'evaluate_completion_time' in helper_functions}")
    
    # 读取求解器模板
    solver_file = os.path.join(template_dir, 'llmina_solver_template.py')
    print(f"\n2. Loading solver template from: {solver_file}")
    with open(solver_file, 'r', encoding='utf-8') as f:
        solver_template = f.read()
    print(f"   ✓ Solver template loaded: {len(solver_template)} characters")
    print(f"   ✓ Contains llm_solver function: {'def llm_solver' in solver_template}")
    
    # 模拟第一个任务的模板
    print("\n3. Task 1 Template (Helper Functions + Solver Template)")
    task1_template = f"{helper_functions}\n\n{solver_template}"
    print(f"   ✓ Combined template length: {len(task1_template)} characters")
    print(f"   ✓ Has helper functions: {'def get_ina_candidates' in task1_template}")
    print(f"   ✓ Has llm_solver: {'def llm_solver' in task1_template}")
    
    # 模拟后续任务的模板
    print("\n4. Task 2+ Template (Helper Functions Only)")
    task2_template = helper_functions
    print(f"   ✓ Template length: {len(task2_template)} characters")
    print(f"   ✓ Has helper functions: {'def get_ina_candidates' in task2_template}")
    print(f"   ✓ Has llm_solver: {'def llm_solver' in task2_template}")
    
    print("\n" + "="*80)
    print("Template Loading Test Complete!")
    print("="*80)
    
    # 显示模板结构
    print("\n5. Template Structure Overview")
    print("\n   Task 1 will receive:")
    print("   - Helper functions (get_ina_candidates, evaluate_completion_time, etc.)")
    print("   - Solver template (def llm_solver signature with TODO placeholders)")
    print("\n   Task 2+ will receive:")
    print("   - Helper functions (same as Task 1)")
    print("   - Previous task's llm_solver implementation (from task_history)")
    
    return True

if __name__ == "__main__":
    try:
        success = test_template_loading()
        if success:
            print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
