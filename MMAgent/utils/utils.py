import json
from typing import Dict
import os
import yaml
from datetime import datetime


def read_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def read_json_file(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def write_text_file(file_path: str, content: str):
    # 确保父目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def write_json_file(file_path: str, data:dict) -> Dict:
    # 确保父目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as json_file:
        json_file.write(json.dumps(data, indent=4, ensure_ascii=False))


def parse_llm_output_to_json(output_text: str) -> dict:
    """
    Safely parse LLM output text into a Python dictionary.
    """
    start = output_text.find("{")
    end = output_text.rfind("}") + 1
    json_str = output_text[start:end]
    try:
        data = json.loads(json_str)
    except:
        raise
        data = {}
    return data

def json_to_markdown(paper):
    """
    Converts a paper dictionary to a Markdown string with multi-level headlines.

    Args:
        paper (dict): The paper dictionary containing problem details and tasks.

    Returns:
        str: A Markdown-formatted string representing the paper.
    """
    markdown_lines = []

    # Problem Background
    markdown_lines.append("## Problem Background")
    markdown_lines.append(paper.get('problem_background', 'No background provided.') + "\n")

    # Problem Requirement
    markdown_lines.append("## Problem Requirement")
    markdown_lines.append(paper.get('problem_requirement', 'No requirements provided.') + "\n")

    # Problem Analysis
    markdown_lines.append("## Problem Analysis")
    markdown_lines.append(paper.get('problem_analysis', 'No analysis provided.') + "\n")

    # Problem Modeling
    if 'problem_modeling' in paper:
        markdown_lines.append("## Problem Modeling")
        markdown_lines.append(paper.get('problem_modeling', 'No modeling provided.') + "\n")

    # Tasks
    tasks = paper.get('tasks', [])
    if tasks:
        markdown_lines.append("## Tasks\n")
        for idx, task in enumerate(tasks, start=1):

            markdown_lines.append(f"### Task {idx}")

            task_description = task.get('task_description', 'No description provided.')
            markdown_lines.append("#### Task Description")
            markdown_lines.append(task_description + "\n")

            # Task Analysis
            task_analysis = task.get('task_analysis', 'No analysis provided.')
            markdown_lines.append("#### Task Analysis")
            markdown_lines.append(task_analysis + "\n")

            # Mathematical Formulas
            task_formulas = task.get('mathematical_formulas', 'No formulas provided.')
            markdown_lines.append("#### Mathematical Formulas")
            if isinstance(task_formulas, list):
                for formula in task_formulas:
                    markdown_lines.append(f"$${formula}$$")
            else:
                markdown_lines.append(f"$${task_formulas}$$")
            markdown_lines.append("")  # Add an empty line

            # Mathematical Modeling Process
            task_modeling = task.get('mathematical_modeling_process', 'No modeling process provided.')
            markdown_lines.append("#### Mathematical Modeling Process")
            markdown_lines.append(task_modeling + "\n")

            # Result
            task_result = task.get('result', 'No result provided.')
            markdown_lines.append("#### Result")
            markdown_lines.append(task_result + "\n")

            # Answer
            task_answer = task.get('answer', 'No answer provided.')
            markdown_lines.append("#### Answer")
            markdown_lines.append(task_answer + "\n")

            # Charts
            charts = task.get('charts', [])
            if charts:
                markdown_lines.append("#### Charts")
                for i, chart in enumerate(charts, start=1):
                    markdown_lines.append(f"##### Chart {i}")
                    markdown_lines.append(chart + "\n")

    # Combine all lines into a single string
    markdown_str = "\n".join(markdown_lines)
    return markdown_str


def json_to_markdown_general(json_data):
    """
    Convert a JSON object to a markdown format.

    Args:
    - json_data (str or dict): The JSON data to convert. It can be a JSON string or a dictionary.

    Returns:
    - str: The markdown formatted string.
    """
    
    if isinstance(json_data, str):
        json_data = json.loads(json_data)  # If input is a JSON string, parse it.

    def recursive_markdown(data, indent=0):
        markdown_str = ""
        indent_space = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                markdown_str += f"### {key}\n"
                markdown_str += recursive_markdown(value, indent + 1)
        elif isinstance(data, list):
            for index, item in enumerate(data):
                markdown_str += f"- **Item {index + 1}**\n"
                markdown_str += recursive_markdown(item, indent + 1)
        else:
            markdown_str += f"- {data}\n"
        
        return markdown_str
    
    markdown = recursive_markdown(json_data)
    return markdown


def save_solution(solution, name, path):
    write_json_file(f'{path}/json/{name}.json', solution)
    markdown_str = json_to_markdown(solution)
    write_text_file(f'{path}/markdown/{name}.md', markdown_str)


def mkdir(path):
    # Create base path and all subdirectories if they do not exist
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'json'), exist_ok=True)
    os.makedirs(os.path.join(path, 'markdown'), exist_ok=True)
    os.makedirs(os.path.join(path, 'latex'), exist_ok=True)
    os.makedirs(os.path.join(path, 'code'), exist_ok=True)
    os.makedirs(os.path.join(path, 'usage'), exist_ok=True)


def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # '/home/hyn/LLM-MM-Agent-LLMINA-clean'
    
    config_abs_path = os.path.join(project_root, 'config.yaml') # '/home/hyn/LLM-MM-Agent-LLMINA-clean/config.yaml'
    with open(config_abs_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config

def extract_function_from_code(code: str, function_name: str) -> tuple:
    """
    使用 AST 从完整代码中鲁棒地提取指定函数的代码，避免匹配到注释或字符串中的伪定义。
    
    Args:
        code: 完整的代码字符串
        function_name: 要提取的函数名
    
    Returns:
        (函数代码, 起始行号, 结束行号, 缩进字符串)
        如果未找到，返回 (None, -1, -1, "")
    """
    # 1. 尝试使用 AST 解析寻找准确的函数定义
    ast_success = False
    try:
        tree = ast.parse(code)
        ast_success = True # 标记 AST 解析成功
        target_node = None
        
        # 遍历 AST 寻找匹配的 FunctionDef
        candidates = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                candidates.append(node)
        
        if candidates:
            target_node = candidates[-1] # 选择最后一个匹配项
            
            # 使用 AST 定位起始行
            start_line_idx = target_node.lineno - 1
            if target_node.decorator_list:
                 start_line_idx = min(d.lineno for d in target_node.decorator_list) - 1
            
            lines = code.split('\n')
            
            # 获取缩进
            def_line = lines[target_node.lineno - 1]
            indent_match = re.match(r'^(\s*)def', def_line)
            indent_prefix = indent_match.group(1) if indent_match else ''
            
            indent_level = len(indent_prefix)
            end_line_idx = len(lines)
            
            # 确定结束行
            for i in range(target_node.lineno, len(lines)):
                line = lines[i]
                stripped = line.strip()
                if stripped:
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_level and not line.lstrip().startswith('#'):
                        end_line_idx = i
                        break
            
            function_code = '\n'.join(lines[start_line_idx:end_line_idx])
            return function_code, start_line_idx, end_line_idx, indent_prefix
        
        # [Critical Fix] 
        # 如果 AST 解析成功但没找到函数，说明该函数真的不存在（或者在注释/字符串中）。
        # 此时绝对不能回退到 Regex，否则Regex会错误地匹配到注释里的代码。
        return None, -1, -1, ""

    except SyntaxError:
        # 只有在代码本身有语法错误导致 AST 无法解析时，才回退到 Regex
        pass
    except Exception as e:
        print(f"[Warning] AST extraction failed for {function_name}: {e}. Fallback to regex.")
    
    # 2. 如果 AST 失败（例如代码本身有语法错误），回退到 Regex 方式（兼容旧逻辑）
    lines = code.split('\n')
    
    # 查找函数定义行（排除注释和字符串内的内容比较难，只能通过行首匹配尝试）
    func_pattern = rf'^\s*def\s+{re.escape(function_name)}\s*\('
    start_line = -1
    indent_level = -1
    indent_prefix = ''
    
    for i, line in enumerate(lines):
        if re.match(func_pattern, line):
            # 再次检查是否是注释（简单检查）
            if not line.strip().startswith('#'):
                start_line = i
                indent_level = len(line) - len(line.lstrip())
                indent_prefix = line[:indent_level]
                break
    
    if start_line == -1:
        return None, -1, -1, ''
    
    end_line = len(lines)
    for i in range(start_line + 1, len(lines)):
        line = lines[i]
        if line.strip():
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level:
                end_line = i
                break
    
    function_code = '\n'.join(lines[start_line:end_line])
    return function_code, start_line, end_line, indent_prefix

def format_function_code_with_indent(new_function_code: str, indent_prefix: str) -> str:
    """
    将新的函数代码去除多余缩进并应用目标缩进
    """
    stripped = new_function_code.strip('\n')
    if not stripped:
        return ''
    dedented = textwrap.dedent(stripped)
    lines = dedented.split('\n')
    if not indent_prefix:
        return '\n'.join(lines)
    formatted_lines = [f"{indent_prefix}{line}" if line.strip() else '' for line in lines]
    return '\n'.join(formatted_lines)

def replace_function_in_code(original_code: str, function_name: str, new_function_code: str) -> str:
    """
    替换代码中的指定函数并保持缩进一致
    
    Args:
        original_code: 原始完整代码
        function_name: 要替换的函数名
        new_function_code: 新的函数代码
    
    Returns:
        替换后的完整代码
    """
    lines = original_code.split('\n')
    old_func_code, start_line, end_line, indent_prefix = extract_function_from_code(original_code, function_name)
    
    if old_func_code is None:
        print(f"[Warning] Function {function_name} not found in code, appending...")
        formatted_code = textwrap.dedent(new_function_code.strip('\n'))
        return original_code + '\n\n' + formatted_code

    formatted_code = format_function_code_with_indent(new_function_code, indent_prefix)
    new_lines = lines[:start_line] + formatted_code.split('\n') + lines[end_line:]
    return '\n'.join(new_lines)
