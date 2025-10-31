#!/bin/bash
# 使用 LangGraph 框架求解 LLMINA 问题的测试脚本

echo "======================================================================"
echo "LangGraph 框架 LLMINA 问题求解测试"
echo "======================================================================"
echo ""

# 激活 conda 环境
echo "1. 激活 agent_test 环境..."
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate agent_test

# 设置参数
MODEL_NAME="qwen3-30b-a3b"
API_KEY="any"  # 替换为实际的 API key
PROBLEM_PATH="MMBench/problem/LLMINA.json"
OUTPUT_DIR="MMAgent/output/LLMINA-LangGraph-Test"

echo "2. 配置参数:"
echo "   模型: $MODEL_NAME"
echo "   问题: $PROBLEM_PATH"
echo "   输出: $OUTPUT_DIR"
echo ""

# 运行求解器
echo "3. 运行 LangGraph 求解器..."
echo "======================================================================"

conda run -n agent_test python MMAgent/run_llmina_langgraph.py \
    --model_name "$MODEL_NAME" \
    --key "$API_KEY" \
    --task LLMINA \
    --clarification_rounds 1 \
    --modeling_rounds 1 \
    --task_num 2

echo ""
echo "======================================================================"
echo "测试完成"
echo "======================================================================"
echo ""
echo "检查输出目录:"
echo "  ls -lh $OUTPUT_DIR/usage/"
echo ""
echo "查看 LLM 使用统计:"
echo "  cat $OUTPUT_DIR/usage/LLMINA.json"
echo ""
echo "查看运行时间:"
echo "  cat $OUTPUT_DIR/usage/runtime.txt"
