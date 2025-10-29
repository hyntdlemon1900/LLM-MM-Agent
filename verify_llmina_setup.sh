#!/bin/bash
# LLMINA工程基础验证脚本

echo "========================================="
echo "LLMINA工程基础验证"
echo "========================================="
echo ""

# 检查Python版本
echo "[1/6] 检查Python版本..."
python --version
if [ $? -ne 0 ]; then
    echo "❌ Python未安装或不在PATH中"
    exit 1
fi
echo "✓ Python检查通过"
echo ""

# 检查关键文件
echo "[2/6] 检查关键文件..."
files=(
    "MMAgent/main_llmina.py"
    "MMAgent/utils/llmina_pipeline.py"
    "MMAgent/agent/llmina_modeling.py"
    "MMAgent/agent/llmina_algorithm.py"
    "MMAgent/agent/llmina_code_generator.py"
    "MMAgent/test_llmina_solver.py"
    "MMBench/problem/LLMINA.json"
    "config.yaml"
)

for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 缺少文件: $file"
        exit 1
    fi
done
echo "✓ 所有关键文件存在"
echo ""

# 检查Python模块导入
echo "[3/6] 检查Python模块导入..."
python -c "
import sys
import os
sys.path.insert(0, 'MMAgent')

try:
    from agent.llmina_modeling import LLMINAModeling
    from agent.llmina_algorithm import LLMINAAlgorithmDesigner
    from agent.llmina_code_generator import LLMINACodeGenerator
    from utils.llmina_pipeline import llmina_solver_pipeline
    print('✓ 所有LLMINA模块导入成功')
except ImportError as e:
    print(f'❌ 模块导入失败: {e}')
    sys.exit(1)
"
if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# 检查配置文件
echo "[4/6] 检查配置文件..."
python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    if 'llmina' in config:
        print('✓ LLMINA配置存在')
        print(f\"  - problem_analysis_round: {config['llmina'].get('problem_analysis_round', 'N/A')}\")
        print(f\"  - algorithm_design_round: {config['llmina'].get('algorithm_design_round', 'N/A')}\")
        print(f\"  - code_generation_round: {config['llmina'].get('code_generation_round', 'N/A')}\")
    else:
        print('❌ LLMINA配置不存在')
        sys.exit(1)
"
if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# 检查LLMINA问题文件
echo "[5/6] 检查LLMINA问题文件..."
python -c "
import json
with open('MMBench/problem/LLMINA.json', 'r') as f:
    problem = json.load(f)
    required_keys = ['background', 'problem_requirement', 'problem_formulation', 'variable_description']
    missing = [k for k in required_keys if k not in problem]
    if missing:
        print(f'❌ LLMINA.json缺少键: {missing}')
        sys.exit(1)
    else:
        print('✓ LLMINA.json格式正确')
        print(f'  - 包含 {len(problem.get(\"variable_description\", {}))} 个变量定义')
        print(f'  - 包含 {len(problem.get(\"problem_formulation\", {}).get(\"key_constraints\", []))} 个关键约束')
"
if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# 检查solver模板
echo "[6/6] 检查solver模板..."
python -c "
import ast
with open('MMAgent/code_template/llmina_solver_template.py', 'r') as f:
    code = f.read()
    try:
        ast.parse(code)
        print('✓ Solver模板语法正确')
        if 'def llm_solver' in code:
            print('  - 包含llm_solver函数定义')
        else:
            print('❌ 缺少llm_solver函数定义')
            sys.exit(1)
    except SyntaxError as e:
        print(f'❌ Solver模板语法错误: {e}')
        sys.exit(1)
"
if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

echo "========================================="
echo "✓ 所有基础验证通过！"
echo "========================================="
echo ""
echo "下一步："
echo "1. 运行生成器: cd MMAgent && python main_llmina.py --task LLMINA"
echo "2. 查看文档: cat LLMINA_QUICKSTART.md"
echo "3. 测试solver: python MMAgent/test_llmina_solver.py <solver_path>"
echo ""
