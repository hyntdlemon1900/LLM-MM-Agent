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

def main():
    """简化的主函数"""
    parser = argparse.ArgumentParser(description='LLM Solver with Checkpoint Support')
    parser.add_argument('--task', type=str, default='LLMINA', help='Task name (use LLMINA for the specialized solver)')
    parser.add_argument('--checkpoint', action='store_true', help='Enable checkpoints (overrides config)')
    parser.add_argument('--resume', type=str, metavar='THREAD_ID', help='Resume from specific thread ID')
    parser.add_argument('--list-checkpoints', action='store_true', help='List all available checkpoints and exit')
    args = parser.parse_args()
    
    # 获取配置（从 config.yaml）
    config = load_config()
    checkpoint_config = config.get('checkpoint', {})
    
    # 处理列出检查点请求
    if args.list_checkpoints:
        # sys.path.insert(0, str(Path(__file__).parent.parent)) # 表示将新的路径插入到 sys.path 的最前面
        try:
            from scripts.checkpoint_manager import CheckpointManager
            db_path = checkpoint_config.get('path', './checkpoints/llmina.db')
            manager = CheckpointManager(db_path)
            threads = manager.list_threads()
            
            if not threads:
                print("\n" + "="*80)
                print("No checkpoints found")
                print("="*80)
                print(f"Database: {db_path}")
                print("\nRun with --checkpoint to create checkpoints.")
                print("="*80 + "\n")
            else:
                print("\n" + "="*80)
                print(f"Available Checkpoints ({len(threads)} threads)")
                print("="*80)
                print(f"Database: {db_path}\n")
                
                for i, thread in enumerate(threads, 1):
                    print(f"{i}. Thread ID: {thread['thread_id']}")
                    print(f"   Checkpoints: {thread['checkpoint_count']}")
                    print(f"   Last Update: {thread['last_checkpoint']}")
                    print()
                
                print("="*80)
                print(f"\nTo resume: python main_langgraph.py --resume <THREAD_ID>")
                print("="*80 + "\n")
            
            manager.close()
        except Exception as e:
            print(f"Error listing checkpoints: {e}")
    
    # 确定是否启用检查点

    enable_checkpoints = args.checkpoint

    # 构建输出目录：output/{task}_{datetime}
    base_output_dir = config.get('paths').get('output')
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = Path(base_output_dir) / f"{args.task}_{timestamp}" # ./output/LLMINA_20251111-220217
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成或使用提供的 thread_id
    if args.resume:
        thread_id = args.resume
        print(f"\n{'='*80}")
        print(f"📂 Resuming from checkpoint")
        print(f"{'='*80}")
        print(f"Thread ID: {thread_id}")
        print(f"{'='*80}\n")
    else:
        thread_id = f"llmina-{args.task}-{timestamp}" # 'llmina-LLMINA-20251111-220217'
        if enable_checkpoints:
            print(f"\n{'='*80}")
            print(f"💾 Checkpoint enabled")
            print(f"{'='*80}")
            print(f"Thread ID: {thread_id}")
            print(f"Database: {checkpoint_config.get('path', './checkpoints/llmina.db')}") # "./checkpoints/llmina.db"
            print(f"\nTo resume if interrupted:")
            print(f"  python main_langgraph.py --resume {thread_id}")
            print(f"{'='*80}\n")

    # 初始化 LLM
    llm = LLM(
        model_name=config.get('model_name', 'qwen3-30b-a3b'),
        key=config.get('api_key', 'any'),
        api_base=config.get('api_base', 'http://192.168.1.44:8021/v1')
    )
    
    # 构建工作流
    workflow = build_workflow(
        enable_checkpoints=enable_checkpoints,
        checkpoint_path=checkpoint_config.get('path'), # "./checkpoints/llmina.db" 
    )

    # 创建初始状态（所有解析和构造都在内部完成）
    initial_state = create_agent_state(
        llm=llm,
        config=config,
        task_name=args.task,
        output_dir=str(output_dir),
    )
    
    start = time.time()
    
    try:
        # 执行工作流 - 带 thread_id 支持
        if enable_checkpoints:
            final_state = workflow.invoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}}
            )
        else:
            final_state = workflow.invoke(initial_state)
        
        # 提取结果
        solver_path = final_state.get('solver_code_path', '')
        errors = final_state.get('errors', [])
        evaluation_results = final_state.get('evaluation_results', {})
        
        print(f"\n{'='*80}")
        print(f"✓ Completed in {time.time() - start:.2f}s")
        print(f"{'='*80}")
        print(f"Solver: {solver_path}")
        print(f"Errors: {len(errors)}")
        if enable_checkpoints:
            print(f"Thread ID: {thread_id}")
        
        if evaluation_results.get('evaluation_success'):
            print(f"\n✓ Evaluation: Success")
            perf_summary = evaluation_results.get('performance_summary', {})
            for config_key, metrics in perf_summary.items():
                print(f"  {config_key}: makespan={metrics['average_makespan']:.4f}, "
                      f"time={metrics['average_solve_time']:.2f}s")
        elif 'error' in evaluation_results:
            print(f"\n✗ Evaluation: Failed - {evaluation_results['error']}")
        
        print(f"{'='*80}\n")
        
        # 保存结果
        write_json_file(f'{output_dir}/usage/{args.task}.json', llm.get_total_usage())
        write_json_file(f'{output_dir}/usage/{args.task}_result.json', {
            'status': 'success' if solver_path else 'failed',
            'solver_path': solver_path,
            'thread_id': thread_id if enable_checkpoints else None,
            'errors': errors,
            'evaluation_results': evaluation_results,
            'llm_usage': llm.get_total_usage()
        })
        
        with open(f'{output_dir}/usage/runtime.txt', 'w') as f:
            f.write(f"{time.time() - start:.2f}s")
        
        return solver_path
    
    # except KeyboardInterrupt:
    #     print(f"\n{'='*80}")
    #     print(f"⚠️  Interrupted by user")
    #     print(f"{'='*80}")
    #     if enable_checkpoints:
    #         print(f"Progress saved to checkpoint: {thread_id}")
    #         print(f"\nTo resume:")
    #         print(f"  python main_langgraph.py --resume {thread_id}")
    #     else:
    #         print(f"Note: Checkpoints were not enabled. Use --checkpoint to enable.")
    #     print(f"{'='*80}\n")
    #     return None
    
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ Error: {e}")
        print(f"{'='*80}")
        if enable_checkpoints:
            print(f"Progress saved to checkpoint: {thread_id}")
            print(f"\nTo resume:")
            print(f"  python main_langgraph.py --resume {thread_id}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        raise        

if __name__ == "__main__":
    main()
