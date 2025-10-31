"""
检查点和持久化模块
实现状态持久化、断点续传和历史回溯功能
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os
import pickle
import sqlite3
from pathlib import Path

from .state import AgentState, WorkflowMetadata


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点目录
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self.db_path = self.checkpoint_dir / "checkpoints.db"
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 创建检查点表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow_id TEXT NOT NULL,
            workflow_name TEXT,
            node_name TEXT,
            timestamp TEXT,
            state_path TEXT,
            metadata TEXT
        )
        ''')
        
        # 创建工作流元数据表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS workflow_metadata (
            workflow_id TEXT PRIMARY KEY,
            workflow_name TEXT,
            start_time TEXT,
            end_time TEXT,
            status TEXT,
            total_nodes INTEGER,
            completed_nodes INTEGER,
            current_node TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_checkpoint(
        self,
        workflow_id: str,
        node_name: str,
        state: AgentState,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        保存检查点
        
        Args:
            workflow_id: 工作流 ID
            node_name: 节点名称
            state: 状态对象
            metadata: 额外元数据
            
        Returns:
            检查点 ID
        """
        timestamp = datetime.now().isoformat()
        
        # 保存状态到文件
        state_filename = f"{workflow_id}_{node_name}_{timestamp.replace(':', '-')}.pkl"
        state_path = self.checkpoint_dir / state_filename
        
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
        
        # 保存到数据库
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO checkpoints (workflow_id, workflow_name, node_name, timestamp, state_path, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            workflow_id,
            state.get('name', 'Unknown'),
            node_name,
            timestamp,
            str(state_path),
            json.dumps(metadata or {})
        ))
        
        checkpoint_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"[Checkpoint] Saved: {workflow_id} @ {node_name}")
        
        return str(checkpoint_id)
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[AgentState]:
        """
        加载检查点
        
        Args:
            checkpoint_id: 检查点 ID
            
        Returns:
            状态对象
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT state_path FROM checkpoints WHERE id = ?
        ''', (checkpoint_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            print(f"[Checkpoint] Not found: {checkpoint_id}")
            return None
        
        state_path = Path(result[0])
        
        if not state_path.exists():
            print(f"[Checkpoint] State file not found: {state_path}")
            return None
        
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        
        print(f"[Checkpoint] Loaded: {checkpoint_id}")
        
        return state
    
    def load_latest_checkpoint(self, workflow_id: str) -> Optional[AgentState]:
        """
        加载最新的检查点
        
        Args:
            workflow_id: 工作流 ID
            
        Returns:
            状态对象
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, state_path FROM checkpoints 
        WHERE workflow_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
        ''', (workflow_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            print(f"[Checkpoint] No checkpoints found for workflow: {workflow_id}")
            return None
        
        checkpoint_id, state_path = result
        state_path = Path(state_path)
        
        if not state_path.exists():
            print(f"[Checkpoint] State file not found: {state_path}")
            return None
        
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        
        print(f"[Checkpoint] Loaded latest: {workflow_id} (ID: {checkpoint_id})")
        
        return state
    
    def list_checkpoints(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出检查点
        
        Args:
            workflow_id: 工作流 ID（可选，如果为 None 则列出所有）
            
        Returns:
            检查点列表
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if workflow_id:
            cursor.execute('''
            SELECT id, workflow_id, workflow_name, node_name, timestamp, metadata
            FROM checkpoints
            WHERE workflow_id = ?
            ORDER BY timestamp DESC
            ''', (workflow_id,))
        else:
            cursor.execute('''
            SELECT id, workflow_id, workflow_name, node_name, timestamp, metadata
            FROM checkpoints
            ORDER BY timestamp DESC
            ''')
        
        results = cursor.fetchall()
        conn.close()
        
        checkpoints = []
        for row in results:
            checkpoints.append({
                'id': row[0],
                'workflow_id': row[1],
                'workflow_name': row[2],
                'node_name': row[3],
                'timestamp': row[4],
                'metadata': json.loads(row[5])
            })
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str):
        """
        删除检查点
        
        Args:
            checkpoint_id: 检查点 ID
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 获取状态文件路径
        cursor.execute('SELECT state_path FROM checkpoints WHERE id = ?', (checkpoint_id,))
        result = cursor.fetchone()
        
        if result:
            state_path = Path(result[0])
            if state_path.exists():
                state_path.unlink()
        
        # 删除数据库记录
        cursor.execute('DELETE FROM checkpoints WHERE id = ?', (checkpoint_id,))
        conn.commit()
        conn.close()
        
        print(f"[Checkpoint] Deleted: {checkpoint_id}")
    
    def cleanup_old_checkpoints(self, workflow_id: str, keep_last: int = 5):
        """
        清理旧检查点
        
        Args:
            workflow_id: 工作流 ID
            keep_last: 保留最近几个检查点
        """
        checkpoints = self.list_checkpoints(workflow_id)
        
        if len(checkpoints) <= keep_last:
            return
        
        to_delete = checkpoints[keep_last:]
        
        for checkpoint in to_delete:
            self.delete_checkpoint(str(checkpoint['id']))
        
        print(f"[Checkpoint] Cleaned up {len(to_delete)} old checkpoints for {workflow_id}")
    
    def save_workflow_metadata(self, metadata: WorkflowMetadata):
        """
        保存工作流元数据
        
        Args:
            metadata: 工作流元数据
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO workflow_metadata 
        (workflow_id, workflow_name, start_time, end_time, status, total_nodes, completed_nodes, current_node)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metadata['workflow_id'],
            metadata['workflow_name'],
            metadata['start_time'],
            metadata.get('end_time'),
            metadata['status'],
            metadata['total_nodes'],
            metadata['completed_nodes'],
            metadata.get('current_node')
        ))
        
        conn.commit()
        conn.close()
    
    def load_workflow_metadata(self, workflow_id: str) -> Optional[WorkflowMetadata]:
        """
        加载工作流元数据
        
        Args:
            workflow_id: 工作流 ID
            
        Returns:
            工作流元数据
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT workflow_id, workflow_name, start_time, end_time, status, 
               total_nodes, completed_nodes, current_node
        FROM workflow_metadata
        WHERE workflow_id = ?
        ''', (workflow_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        return WorkflowMetadata(
            workflow_id=result[0],
            workflow_name=result[1],
            start_time=result[2],
            end_time=result[3],
            status=result[4],
            total_nodes=result[5],
            completed_nodes=result[6],
            current_node=result[7],
            checkpoint_path=None
        )


# ============ 全局检查点管理器 ============

_global_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager(checkpoint_dir: str = "./checkpoints") -> CheckpointManager:
    """获取全局检查点管理器"""
    global _global_checkpoint_manager
    
    if _global_checkpoint_manager is None:
        _global_checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    return _global_checkpoint_manager


def save_checkpoint(
    workflow_id: str,
    node_name: str,
    state: AgentState,
    metadata: Optional[Dict[str, Any]] = None,
    checkpoint_dir: str = "./checkpoints"
) -> str:
    """便捷函数：保存检查点"""
    manager = get_checkpoint_manager(checkpoint_dir)
    return manager.save_checkpoint(workflow_id, node_name, state, metadata)


def load_checkpoint(
    checkpoint_id: str,
    checkpoint_dir: str = "./checkpoints"
) -> Optional[AgentState]:
    """便捷函数：加载检查点"""
    manager = get_checkpoint_manager(checkpoint_dir)
    return manager.load_checkpoint(checkpoint_id)


def load_latest_checkpoint(
    workflow_id: str,
    checkpoint_dir: str = "./checkpoints"
) -> Optional[AgentState]:
    """便捷函数：加载最新检查点"""
    manager = get_checkpoint_manager(checkpoint_dir)
    return manager.load_latest_checkpoint(workflow_id)
