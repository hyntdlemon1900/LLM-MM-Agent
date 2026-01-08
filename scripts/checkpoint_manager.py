"""
检查点管理工具

功能：
- 列出所有检查点
- 查看检查点详情
- 清理旧检查点
- 导出/恢复检查点
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sqlite3


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, db_path: str):
        """
        初始化管理器
        
        Args:
            db_path: SQLite 数据库路径
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            print(f"⚠️  Database not found: {self.db_path}")
            print(f"No checkpoints available yet.")
            self.conn = None
        else:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row
    
    def list_threads(self) -> List[Dict[str, Any]]:
        """列出所有线程"""
        if not self.conn:
            return []
        
        cursor = self.conn.execute("""
            SELECT 
                thread_id,
                COUNT(*) as checkpoint_count,
                MIN(created_at) as first_checkpoint,
                MAX(created_at) as last_checkpoint
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY last_checkpoint DESC
        """)
        
        threads = []
        for row in cursor:
            threads.append({
                'thread_id': row['thread_id'],
                'checkpoint_count': row['checkpoint_count'],
                'first_checkpoint': row['first_checkpoint'],
                'last_checkpoint': row['last_checkpoint']
            })
        
        return threads
    
    def get_thread_checkpoints(self, thread_id: str) -> List[Dict[str, Any]]:
        """获取指定线程的所有检查点"""
        if not self.conn:
            return []
        
        cursor = self.conn.execute("""
            SELECT 
                checkpoint_id,
                parent_checkpoint_id,
                checkpoint_ns,
                metadata,
                created_at
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY created_at ASC
        """, (thread_id,))
        
        checkpoints = []
        for row in cursor:
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            checkpoints.append({
                'checkpoint_id': row['checkpoint_id'],
                'parent_checkpoint_id': row['parent_checkpoint_id'],
                'checkpoint_ns': row['checkpoint_ns'],
                'metadata': metadata,
                'created_at': row['created_at'],
                'step': metadata.get('step', '?'),
                'source': metadata.get('source', '?')
            })
        
        return checkpoints
    
    def get_checkpoint_details(self, thread_id: str, checkpoint_id: str = None) -> Dict[str, Any]:
        """获取检查点详细信息"""
        if not self.conn:
            return {}
        
        if checkpoint_id is None:
            # 获取最新的检查点
            cursor = self.conn.execute("""
                SELECT checkpoint, metadata
                FROM checkpoints
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (thread_id,))
        else:
            cursor = self.conn.execute("""
                SELECT checkpoint, metadata
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_id = ?
            """, (thread_id, checkpoint_id))
        
        row = cursor.fetchone()
        if not row:
            return {}
        
        # 反序列化检查点数据
        checkpoint_data = pickle.loads(row['checkpoint'])
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        
        # 提取关键信息
        channel_values = checkpoint_data.get('channel_values', {})
        
        return {
            'metadata': metadata,
            'step': metadata.get('step', '?'),
            'source': metadata.get('source', '?'),
            'state_keys': list(channel_values.keys()),
            'output_dir': channel_values.get('output_dir'),
            'current_function_id': channel_values.get('current_function_id'),
            'function_count': len(channel_values.get('function_architecture', [])),
            'messages_count': len(channel_values.get('messages', [])),
            'errors_count': len(channel_values.get('errors', [])),
            'solver_code_path': channel_values.get('solver_code_path'),
            'evaluation_success': channel_values.get('evaluation_results', {}).get('evaluation_success')
        }
    
    def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """清理旧检查点"""
        if not self.conn:
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        # 查找要删除的线程
        cursor = self.conn.execute("""
            SELECT DISTINCT thread_id
            FROM checkpoints
            WHERE created_at < ?
        """, (cutoff_str,))
        
        threads_to_delete = [row[0] for row in cursor]
        
        if not threads_to_delete:
            return 0
        
        # 删除旧检查点
        for thread_id in threads_to_delete:
            self.conn.execute("""
                DELETE FROM checkpoints
                WHERE thread_id = ?
            """, (thread_id,))
            
            self.conn.execute("""
                DELETE FROM writes
                WHERE thread_id = ?
            """, (thread_id,))
        
        self.conn.commit()
        
        return len(threads_to_delete)
    
    def cleanup_except_latest(self, keep_count: int = 10) -> int:
        """只保留最新的 N 个检查点"""
        if not self.conn:
            return 0
        
        threads = self.list_threads()
        
        if len(threads) <= keep_count:
            return 0
        
        # 删除旧的线程
        threads_to_delete = threads[keep_count:]
        
        for thread in threads_to_delete:
            thread_id = thread['thread_id']
            self.conn.execute("""
                DELETE FROM checkpoints
                WHERE thread_id = ?
            """, (thread_id,))
            
            self.conn.execute("""
                DELETE FROM writes
                WHERE thread_id = ?
            """, (thread_id,))
        
        self.conn.commit()
        
        return len(threads_to_delete)
    
    def export_checkpoint(self, thread_id: str, output_file: str):
        """导出检查点到文件"""
        if not self.conn:
            print("No database connection")
            return
        
        # 导出所有相关数据
        export_data = {
            'thread_id': thread_id,
            'checkpoints': [],
            'writes': []
        }
        
        # 导出检查点
        cursor = self.conn.execute("""
            SELECT * FROM checkpoints
            WHERE thread_id = ?
        """, (thread_id,))
        
        for row in cursor:
            export_data['checkpoints'].append(dict(row))
        
        # 导出写入记录
        cursor = self.conn.execute("""
            SELECT * FROM writes
            WHERE thread_id = ?
        """, (thread_id,))
        
        for row in cursor:
            export_data['writes'].append(dict(row))
        
        # 保存到文件
        with open(output_file, 'wb') as f:
            pickle.dump(export_data, f)
        
        print(f"✓ Exported checkpoint to: {output_file}")
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()


def format_datetime(dt_str: str) -> str:
    """格式化日期时间"""
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return dt_str


def cmd_list(args):
    """列出所有检查点"""
    manager = CheckpointManager(args.db)
    threads = manager.list_threads()
    
    if not threads:
        print("\n" + "="*80)
        print("No checkpoints found")
        print("="*80)
        print(f"Database: {args.db}")
        print("\nRun with --checkpoint to create checkpoints.")
        print("="*80 + "\n")
        return
    
    print("\n" + "="*80)
    print(f"Available Checkpoints ({len(threads)} threads)")
    print("="*80)
    print(f"Database: {args.db}\n")
    
    for i, thread in enumerate(threads, 1):
        print(f"{i}. Thread ID: {thread['thread_id']}")
        print(f"   Checkpoints: {thread['checkpoint_count']}")
        print(f"   First: {format_datetime(thread['first_checkpoint'])}")
        print(f"   Last:  {format_datetime(thread['last_checkpoint'])}")
        print()
    
    print("="*80)
    print(f"\nTo resume: python main_langgraph.py --resume <THREAD_ID>")
    print(f"To view details: python scripts/checkpoint_manager.py show --thread-id <THREAD_ID>")
    print("="*80 + "\n")
    
    manager.close()


def cmd_show(args):
    """显示检查点详情"""
    if not args.thread_id:
        print("Error: --thread-id required for 'show' command")
        return
    
    manager = CheckpointManager(args.db)
    
    # 显示线程的所有检查点
    checkpoints = manager.get_thread_checkpoints(args.thread_id)
    
    if not checkpoints:
        print(f"\n✗ No checkpoints found for thread: {args.thread_id}\n")
        manager.close()
        return
    
    print("\n" + "="*80)
    print(f"Checkpoints for Thread: {args.thread_id}")
    print("="*80)
    print(f"Total Checkpoints: {len(checkpoints)}\n")
    
    for i, cp in enumerate(checkpoints, 1):
        print(f"{i}. Step {cp['step']} ({cp['source']})")
        print(f"   Checkpoint ID: {cp['checkpoint_id'][:16]}...")
        print(f"   Created: {format_datetime(cp['created_at'])}")
        print()
    
    # 显示最新检查点的详细信息
    print("="*80)
    print("Latest Checkpoint Details")
    print("="*80)
    
    details = manager.get_checkpoint_details(args.thread_id)
    
    if details:
        print(f"Step: {details['step']}")
        print(f"Source: {details['source']}")
        print(f"\nState Information:")
        print(f"  Output Directory: {details['output_dir']}")
        print(f"  Current Function ID: {details['current_function_id']}")
        print(f"  Total Functions: {details['function_count']}")
        print(f"  Messages Count: {details['messages_count']}")
        print(f"  Errors Count: {details['errors_count']}")
        
        if details['solver_code_path']:
            print(f"\n✓ Solver Generated: {details['solver_code_path']}")
        
        if details['evaluation_success'] is not None:
            status = "✓ Success" if details['evaluation_success'] else "✗ Failed"
            print(f"\nEvaluation: {status}")
    
    print("="*80 + "\n")
    
    manager.close()


def cmd_cleanup(args):
    """清理旧检查点"""
    manager = CheckpointManager(args.db)
    
    print("\n" + "="*80)
    print("Cleaning up old checkpoints")
    print("="*80)
    
    if args.keep:
        # 只保留最新的 N 个
        deleted = manager.cleanup_except_latest(args.keep)
        print(f"Deleted {deleted} old threads (kept latest {args.keep})")
    else:
        # 删除 N 天前的
        deleted = manager.cleanup_old_checkpoints(args.days)
        print(f"Deleted {deleted} threads older than {args.days} days")
    
    print("="*80 + "\n")
    
    manager.close()


def cmd_export(args):
    """导出检查点"""
    if not args.thread_id:
        print("Error: --thread-id required for 'export' command")
        return
    
    manager = CheckpointManager(args.db)
    
    output_file = args.output or f"checkpoint_{args.thread_id}.pkl"
    
    print(f"\nExporting checkpoint: {args.thread_id}")
    print(f"Output file: {output_file}")
    
    manager.export_checkpoint(args.thread_id, output_file)
    
    manager.close()


def main():
    parser = argparse.ArgumentParser(
        description='LangGraph Checkpoint Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all checkpoints
  python scripts/checkpoint_manager.py list
  
  # Show details of a specific thread
  python scripts/checkpoint_manager.py show --thread-id llmina-LLMINA-20241111-143022
  
  # Cleanup checkpoints older than 7 days
  python scripts/checkpoint_manager.py cleanup --days 7
  
  # Keep only latest 10 checkpoints
  python scripts/checkpoint_manager.py cleanup --keep 10
  
  # Export a checkpoint
  python scripts/checkpoint_manager.py export --thread-id llmina-LLMINA-20241111-143022
        """
    )
    
    parser.add_argument(
        'command',
        choices=['list', 'show', 'cleanup', 'export'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--db',
        default='./checkpoints/llmina.db',
        help='Path to checkpoint database (default: ./checkpoints/llmina.db)'
    )
    
    parser.add_argument(
        '--thread-id',
        type=str,
        help='Thread ID for show/export commands'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Days threshold for cleanup (default: 7)'
    )
    
    parser.add_argument(
        '--keep',
        type=int,
        help='Number of latest checkpoints to keep (for cleanup)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for export'
    )
    
    args = parser.parse_args()
    
    # 执行命令
    if args.command == 'list':
        cmd_list(args)
    elif args.command == 'show':
        cmd_show(args)
    elif args.command == 'cleanup':
        cmd_cleanup(args)
    elif args.command == 'export':
        cmd_export(args)


if __name__ == "__main__":
    main()
