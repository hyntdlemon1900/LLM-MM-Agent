import os
import requests
import openai
from dotenv import load_dotenv
import json
from langchain_openai import ChatOpenAI

load_dotenv()

class LLM:

    usages = []
    def __init__(self, model_name, key, logger=None, user_id=None, api_base=None):
        self.model_name = model_name
        self.logger = logger
        self.user_id = user_id
        self.api_key = key
        
        # 如果提供了 api_base 则使用，否则根据模型名称设置
        if api_base:
            self.api_base = api_base
        elif self.model_name in ['deepseek-chat', 'deepseek-reasoner']:
            self.api_base = os.getenv('DEEPSEEK_API_BASE')
        elif self.model_name in ['qwen2.5-72b-instruct']:
            self.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif self.model_name in ['gpt-4o', 'gpt-4']:
            self.api_base = os.getenv('OPENAI_API_BASE')
        else:
            # 默认使用本地服务
            self.api_base = "http://192.168.1.44:8021/v1"
        
        if not self.api_key:
            raise ValueError('API key not found in environment variables')

        # 使用 LangChain 的 ChatOpenAI
        self.client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.api_base,
            temperature=0.7,
            max_retries=2
        )

    def reset(self, api_key=None, api_base=None, model_name=None):
        if api_key:
            self.api_key = api_key
        if api_base:
            self.api_base = api_base
        if model_name:
            self.model_name = model_name
        
        # 重新创建 ChatOpenAI 实例
        self.client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.api_base,
            temperature=0.7,
            max_retries=2
        )

    def generate(self, prompt, system='', usage=True):
        try:
            # 构建消息列表
            messages = []
            if system:
                messages.append(('system', system))
            messages.append(('human', prompt))
            
            # 使用 LangChain 的 invoke 方法
            response = self.client.invoke(messages)
            
            # 提取回复内容
            answer = response.content
            
            # 提取使用统计（如果可用）
            usage_data = {
                'completion_tokens': 0,
                'prompt_tokens': 0,
                'total_tokens': 0
            }
            
            # LangChain 的 response 可能包含 usage_metadata
            if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                token_usage = response.response_metadata['token_usage']
                usage_data = {
                    'completion_tokens': token_usage.get('completion_tokens', 0),
                    'prompt_tokens': token_usage.get('prompt_tokens', 0),
                    'total_tokens': token_usage.get('total_tokens', 0)
                }
            elif hasattr(response, 'usage_metadata'):
                # 新版本的 LangChain 可能直接有 usage_metadata 属性
                usage_data = {
                    'completion_tokens': response.usage_metadata.get('output_tokens', 0),
                    'prompt_tokens': response.usage_metadata.get('input_tokens', 0),
                    'total_tokens': response.usage_metadata.get('total_tokens', 0)
                }
            
            if self.logger:
                self.logger.info(f"[LLM] UserID: {self.user_id} Key: {self.api_key}, Model: {self.model_name}, Usage: {usage_data}")
            
            if usage:
                self.usages.append(usage_data)
            
            return answer

        except Exception as e:
            return f'An error occurred: {e}'

    def get_total_usage(self):
        total_usage = { 
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }
        for usage in self.usages:
            for key, value in usage.items():
                total_usage[key] += value
        return total_usage
        
    def clear_usage(self):
        self.usages = []