"""
Core MCTS Implementation for LLMINA Evolution.
"""
import math
import uuid
import copy
from typing import List, Optional, Any, Dict

class MCTSNode:
    def __init__(self, 
                 score: float = 0.0, 
                 parent: Optional['MCTSNode'] = None,
                 description: str = "Root",
                 code_path: str = "",
                 config: Dict = None,
                 operator: str = "i1",
                 node_id: str = None):
        
        if node_id:
            self.node_id = str(node_id)
        else:
            self.node_id = str(uuid.uuid4())[:8]
            
        self.score = score  # External quality score (Objective)
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.Q = 0.0      # Accumulated value (e.g. from rollout or evaluation)
        
        # Meta-data about the algorithm
        self.description = description
        self.code_path = code_path
        self.config = config or {} # Stores 'function_codes', 'architecture', etc.
        self.operator = operator # Operator used to create this node
        
        # Upper Confidence Bound parameters
        self.exploration_weight = 1.0

    def add_child(self, child_node: 'MCTSNode'):
        self.children.append(child_node)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def update(self, reward: float):
        """Update node stats with new reward."""
        self.visits += 1
        # Simple Incremental Mean
        # self.Q += (reward - self.Q) / self.visits 
        # Or standard MCTS max/avg logic?
        # The user's provided implementation uses: 
        # parent.Q = parent.Q * (1 - discount) + best_child_Q * discount
        # But for tree search in optimization (like AlphaGo or evolution), usually Q is average reward.
        # Let's stick to Average Reward for now, or Max Reward if we want optimistic.
        
        # Implementation from mcts.py provided:
        # parent.Q = parent.Q * (1 - discount) + best_child_Q * discount
        # This propagates the BEST child value up.
        
        # For standard UCT, Q is usually average.
        # However, since this is optimization, we care about Finding the Max.
        # Let's update Q to be the Max found in this subtree?
        pass

    def __repr__(self):
        return f"Node(id={self.node_id}, score={self.score:.4f}, Q={self.Q:.4f}, visits={self.visits}, op={self.operator})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dict."""
        return {
            'node_id': self.node_id,
            'score': self.score,
            'parent_id': self.parent.node_id if self.parent else None,
            'children_ids': [child.node_id for child in self.children],
            'visits': self.visits,
            'Q': self.Q,
            'description': self.description,
            'code_path': self.code_path,
            'config': self.config,
            'operator': self.operator
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCTSNode':
        """Deserialize dict to node (links to parent/children must be restored externally)."""
        node = cls(
            score=data.get('score', 0.0),
            description=data.get('description', ''),
            code_path=data.get('code_path', ''),
            config=data.get('config'),
            operator=data.get('operator', 'i1')
        )
        node.node_id = data.get('node_id', str(uuid.uuid4()))
        node.visits = data.get('visits', 0)
        node.Q = data.get('Q', 0.0)
        # parent and children are set by the tree reconstructor
        return node


class MCTS:
    def __init__(self, root_node: MCTSNode = None):
        if root_node:
            self.root = root_node
        else:
            self.root = MCTSNode(description="Root", operator="root")
            
        self.nodes = {self.root.node_id: self.root}
        
        # UCT Parameters
        self.exploration_constant_0 = 0.5 
        
        # Stats
        self.q_min = 0.0
        self.q_max = -10000.0 # Initialize low
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire tree state."""
        nodes_data = {nid: node.to_dict() for nid, node in self.nodes.items()}
        return {
            'nodes': nodes_data,
            'root_id': self.root.node_id,
            'q_min': self.q_min,
            'q_max': self.q_max
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCTS':
        """Reconstruct tree from serialized data."""
        if not data:
            return cls()
            
        nodes_data = data.get('nodes', {})
        root_id = data.get('root_id')
        
        if not nodes_data or not root_id:
            return cls()

        # 1. Create all node instances
        reconstructed_nodes = {}
        for nid, n_data in nodes_data.items():
            reconstructed_nodes[nid] = MCTSNode.from_dict(n_data)
            
        # 2. Re-link parents and children
        for nid, n_data in nodes_data.items():
            node = reconstructed_nodes[nid]
            
            # Link Parent
            parent_id = n_data.get('parent_id')
            if parent_id and parent_id in reconstructed_nodes:
                node.parent = reconstructed_nodes[parent_id]
                
            # Link Children
            children_ids = n_data.get('children_ids', [])
            for child_id in children_ids:
                if child_id in reconstructed_nodes:
                    node.children.append(reconstructed_nodes[child_id])

        # 3. Initialize Tree
        instance = cls()
        instance.nodes = reconstructed_nodes
        if root_id in reconstructed_nodes:
            instance.root = reconstructed_nodes[root_id]
        
        instance.q_min = data.get('q_min', 0.0)
        instance.q_max = data.get('q_max', -10000.0)
        
        return instance
        
    def register_node(self, node: MCTSNode):
        self.nodes[node.node_id] = node

    def select(self) -> MCTSNode:
        """
        Select the most promising node to expand using UCT.
        Starts from root, goes down to a leaf.
        """
        node = self.root
        while not node.is_leaf():
            # If any child has not been visited? Or standard UCT?
            # Standard UCT expects fully expanded node before going deeper.
            # But here "expansion" is expensive (LLM generation).
            # So a node is a candidate for expansion if it's a leaf.
            
            # If we are at a node that has children, we pick the best child according to UCT
            best_child = self.get_best_child(node)
            if best_child:
                node = best_child
            else:
                break
        return node

    def get_best_child(self, node: MCTSNode) -> MCTSNode:
        if not node.children:
            return None
            
        best_score = -float('inf')
        best_children = []
        
        for child in node.children:
            uct_val = self.uct(child)
            if uct_val > best_score:
                best_score = uct_val
                best_children = [child]
            elif uct_val == best_score:
                best_children.append(child)
                
        return random.choice(best_children) if best_children else None

    def uct(self, node: MCTSNode) -> float:
        if node.visits == 0:
            return float('inf')
        
        # Normalize Q
        if self.q_max > self.q_min:
            q_norm = (node.Q - self.q_min) / (self.q_max - self.q_min)
        else:
            q_norm = node.Q
            
        # Exploration term
        parent_visits = node.parent.visits if node.parent else 0
        exploration = self.exploration_constant_0 * math.sqrt(
            math.log(parent_visits + 1) / node.visits
        )
        
        return q_norm + exploration

    def backpropagate(self, node: MCTSNode):
        """
        Update Q-values and visits from node up to root.
        Using the logic from provided mcts.py where Q represents the 'best' potential.
        """
        # First update min/max stats for normalization
        if node.Q < self.q_min: self.q_min = node.Q
        if node.Q > self.q_max: self.q_max = node.Q
        
        curr = node
        while curr:
            curr.visits += 1
            
            # Update Q based on children (Max Q of children)
            # If leaf, Q is its own score (or reward).
            # If internal, max of children.
            if curr.children:
                max_child_q = max(c.Q for c in curr.children)
                # Discount factor from provided code was 1.0, so parent essentially takes max child score
                curr.Q = max_child_q 
            else:
                # Leaf node Q is its own score initially
                # Assuming node.score is set during evaluation
                curr.Q = curr.score
                
            curr = curr.parent
            
import random
