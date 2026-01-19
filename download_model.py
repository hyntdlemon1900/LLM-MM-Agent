from modelscope import snapshot_download

model_dir = snapshot_download(
    'unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF',
    cache_dir='./deepseek_models',
    # 精确匹配 Q5_K_M 文件，避免下载几百 G 的无用文件
    allow_patterns=['*Q5_K_M.gguf']
)
print(f"模型已下载至: {model_dir}")