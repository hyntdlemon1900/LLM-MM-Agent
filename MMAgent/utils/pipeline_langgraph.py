"""
LLMINA问题求解流程（LangGraph版本）
基于langgraph框架重写，模块化各Agent节点，支持工具调用和多轮澄清。
"""

from langgraph.graph import StateGraph
from typing import TypedDict
import sys
import os
# 添加父目录到路径以支持从MMAgent目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llmina_tools import get_ina_candidates, evaluate_completion_time
from agent.llmina_problem_clarification import ProblemClarification
from agent.llmina_summary import ProblemClarificationSummary
from agent.llmina_feedback_judge import ProblemJudge
from agent.llmina_problem_solving import ProblemSolving
from agent.llmina_problem_decompse import ProblemDecompose
from agent.llmina_tools_agent import LLMINAToolsAgent
from prompt.llmina_template import PROBLEM_DESCRIPTION_PROMPT
import os
import json

class PipelineState(TypedDict, total=False):
    problem_str: str
    config: dict
    output_dir: str
    name: str
    clarification_history: list
    clarification_summary: str
    modeling_solution: str
    task_descriptions: list

def make_clarification_node(pc, pj, ps):
    def node(state: dict):
        # 交互式澄清流程
        def dummy_user_input(agent_feedback):
            return ""
        problem_str, history, summary = pc.clarification_actor(pj, ps, state['problem_str'], dummy_user_input)
        new_state = state.copy()
        new_state['problem_str'] = problem_str
        new_state['clarification_history'] = history
        new_state['clarification_summary'] = summary
        return new_state
    return node

def make_solution_node(pu):
    def node(state: dict):
        modeling_solution = pu.solving(state['problem_str'], round=state['config'].get('problem_modeling_round', 2))
        new_state = state.copy()
        new_state['modeling_solution'] = modeling_solution
        return new_state
    return node

def make_decompose_node(pd):
    def node(state: dict):
        # Extract required arguments for decompose_and_refine
        modeling_problem = state['problem_str']
        # If problem_analysis is needed, try to get it from state, else use ''
        problem_analysis = state.get('clarification_summary', '')
        modeling_solution = state.get('modeling_solution', '')
        # Get decomposed_principle from config or use default 'C'
        problem_type = state['config'].get('problem_type', 'C')
        decomposed_principle = problem_type
        tasknum = state['config'].get('tasknum', 3)
        user_prompt = ''
        task_descriptions = pd.decompose_and_refine(
            modeling_problem,
            problem_analysis,
            modeling_solution,
            decomposed_principle,
            tasknum,
            user_prompt
        )
        new_state = state.copy()
        new_state['task_descriptions'] = task_descriptions
        return new_state
    return node

def llmina_solver_pipeline_langgraph(llm, problem_path, config, output_dir, name):
    # 读取问题
    problem = json.load(open(problem_path, 'r'))
    background = problem['background']
    problem_requirement = problem['problem_requirement']
    variable_description = problem['variable_description']
    problem_formulation = problem['problem_formulation']
    code_template = open('/home/hyn/LLM-MM-Agent-LLMINA/MMAgent/code_template/llmina_solver_template.py').read()
    problem_str = PROBLEM_DESCRIPTION_PROMPT.format(
        problem_background=background,
        problem_requirement=problem_requirement,
        variable_description=variable_description,
        problem_formulation=problem_formulation,
        code_template=code_template
    ).strip()

    pc = ProblemClarification(llm)
    pj = ProblemJudge(llm)
    ps = ProblemClarificationSummary(llm)
    pu = ProblemSolving(llm)
    pd = ProblemDecompose(llm)

    graph = StateGraph(PipelineState)
    graph.add_node('clarification', make_clarification_node(pc, pj, ps))
    graph.add_node('solution', make_solution_node(pu))
    graph.add_node('decompose', make_decompose_node(pd))

    graph.add_edge('clarification', 'solution')
    graph.add_edge('solution', 'decompose')
    graph.set_entry_point('clarification')
    graph.set_finish_point('decompose')

    compiled = graph.compile()
    state: PipelineState = {
        'problem_str': problem_str,
        'config': config,
        'output_dir': output_dir,
        'name': name,
        'clarification_history': [],
        'clarification_summary': '',
        'modeling_solution': '',
        'task_descriptions': []
    }
    result = compiled.invoke(state)
    return result
