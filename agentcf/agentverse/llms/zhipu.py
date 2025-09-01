from logging import getLogger
import os
from typing import Dict, List, Optional, Union
import asyncio
from pydantic import BaseModel, Field
import time
import requests
import json

from agentverse.llms.base import LLMResult
from typing import Any

from . import llm_registry
from .base import BaseChatModel, BaseCompletionModel, BaseModelArgs

logger = getLogger()

# 智谱GLM API配置
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")
ZHIPU_API_BASE = os.environ.get("ZHIPU_API_BASE", "https://open.bigmodel.cn/api/paas/v4")

if ZHIPU_API_KEY is None:
    logger.info(
        "ZHIPU API key is not set. Please set the environment variable ZHIPU_API_KEY"
    )
    is_zhipu_available = False
else:
    is_zhipu_available = True

# 尝试导入官方SDK，如果不存在则使用我们的实现
try:
    from zhipu import ZhipuAI, AsyncZhipuAI
    HAS_ZHIPU_SDK = True
except ImportError:
    HAS_ZHIPU_SDK = False
    logger.info("Official Zhipu SDK not found, using custom implementation")


class ZhipuChatArgs(BaseModelArgs):
    model: str = Field(default="glm-4")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)
    n: int = Field(default=1)
    stop: Optional[Union[str, List]] = Field(default=None)
    presence_penalty: float = Field(default=0.0)
    frequency_penalty: float = Field(default=0.0)


class ZhipuCompletionArgs(ZhipuChatArgs):
    model: str = Field(default="glm-4")


class ZhipuEmbeddingArgs(BaseModelArgs):
    model: str = Field(default="embedding-2")


class ZhipuAPI:
    """智谱GLM API客户端"""
    
    def __init__(self, api_key: str, api_base: str = ZHIPU_API_BASE):
        self.api_key = api_key
        self.api_base = api_base
        
        # 优先使用官方SDK
        if HAS_ZHIPU_SDK:
            self.client = ZhipuAI(api_key=api_key)
            self.async_client = AsyncZhipuAI(api_key=api_key)
        else:
            self.client = None
            self.async_client = None
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
    
    def _make_request(self, endpoint: str, data: dict):
        """发送HTTP请求（回退方案）"""
        url = f"{self.api_base}/{endpoint}"
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def chat_completion(self, messages: List[Dict], **kwargs):
        """聊天补全"""
        # 确保消息格式正确
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list) and len(messages) > 0:
            if isinstance(messages[0], str):
                messages = [{"role": "user", "content": messages[0]}]
            elif isinstance(messages[0], list):
                messages = messages[0]
        
        if HAS_ZHIPU_SDK and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=kwargs.get("model", "glm-4"),
                    messages=messages,
                    max_tokens=kwargs.get("max_tokens", 2048),
                    temperature=kwargs.get("temperature", 1.0),
                    top_p=kwargs.get("top_p", 1.0)
                )
                
                # 转换为OpenAI兼容格式
                return {
                    "choices": [{
                        "message": {
                            "content": response.choices[0].message.content,
                            "role": response.choices[0].message.role
                        }
                    }],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            except Exception as e:
                logger.warning(f"Official SDK failed, falling back to HTTP: {e}")
        
        # 回退到HTTP请求
        data = {
            "model": kwargs.get("model", "glm-4"),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 1.0)
        }
        
        # 移除空值参数
        data = {k: v for k, v in data.items() if v is not None}
        
        response = self._make_request("chat/completions", data)
        
        # 确保返回格式与OpenAI兼容
        if "choices" not in response:
            response = {
                "choices": [{
                    "message": {
                        "content": response.get("content", response.get("text", "")),
                        "role": "assistant"
                    }
                }],
                "usage": {
                    "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": response.get("usage", {}).get("total_tokens", 0)
                }
            }
        
        return response
    
    async def async_chat_completion(self, messages: List[Dict], **kwargs):
        """异步聊天补全"""
        # 确保消息格式正确
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list) and len(messages) > 0:
            if isinstance(messages[0], str):
                messages = [{"role": "user", "content": messages[0]}]
            elif isinstance(messages[0], list):
                messages = messages[0]
        
        if HAS_ZHIPU_SDK and self.async_client:
            try:
                response = await self.async_client.chat.completions.create(
                    model=kwargs.get("model", "glm-4"),
                    messages=messages,
                    max_tokens=kwargs.get("max_tokens", 2048),
                    temperature=kwargs.get("temperature", 1.0),
                    top_p=kwargs.get("top_p", 1.0)
                )
                
                # 转换为OpenAI兼容格式
                return {
                    "choices": [{
                        "message": {
                            "content": response.choices[0].message.content,
                            "role": response.choices[0].message.role
                        }
                    }],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            except Exception as e:
                logger.warning(f"Official async SDK failed, falling back to sync: {e}")
        
        # 回退到同步调用
        return self.chat_completion(messages, **kwargs)
    
    def embedding(self, input_text: Union[str, List[str]], **kwargs):
        """文本嵌入"""
        if isinstance(input_text, str):
            input_text = [input_text]
        
        if HAS_ZHIPU_SDK and self.client:
            try:
                response = self.client.embeddings.create(
                    model=kwargs.get("model", "embedding-2"),
                    input=input_text
                )
                
                # 转换为OpenAI兼容格式
                return {
                    "data": [{
                        "embedding": item.embedding
                    } for item in response.data]
                }
            except Exception as e:
                logger.warning(f"Official SDK embedding failed, falling back to HTTP: {e}")
        
        # 回退到HTTP请求
        data = {
            "model": kwargs.get("model", "embedding-2"),
            "input": input_text
        }
        
        response = self._make_request("embeddings", data)
        
        # 确保返回格式与OpenAI兼容
        if "data" not in response:
            response = {
                "data": [{
                    "embedding": response.get("embedding", [])
                }]
            }
        
        return response


@llm_registry.register("glm-4")
@llm_registry.register("glm-3-turbo")
class ZhipuChat(BaseChatModel):
    args: ZhipuChatArgs = Field(default_factory=ZhipuChatArgs)
    api_key_list: list = []
    current_key_idx = 0
    
    def __init__(self, max_retry: int = 3, **kwargs):
        args = ZhipuChatArgs()
        args = args.dict()
        
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        
        super().__init__(args=args, max_retry=max_retry)
        self.api_key_list = kwargs.pop('api_key_list', [])
        if not self.api_key_list:
            self.api_key_list = [ZHIPU_API_KEY] if ZHIPU_API_KEY else []
    
    def _construct_messages(self, prompts: list):
        """构造消息格式"""
        messages = []
        for prompt in prompts:
            if isinstance(prompt, str):
                messages.append([{"role": "user", "content": prompt}])
            elif isinstance(prompt, list):
                messages.append(prompt)
        return messages
    
    def generate_response(self, prompt: str) -> LLMResult:
        """生成响应"""
        messages = self._construct_messages([prompt])
        try:
            api_client = ZhipuAPI(self.api_key_list[self.current_key_idx])
            response = api_client.chat_completion(messages[0], **self.args.dict())
            
            return LLMResult(
                content=response["choices"][0]["message"]["content"],
                send_tokens=response["usage"]["prompt_tokens"],
                recv_tokens=response["usage"]["completion_tokens"],
                total_tokens=response["usage"]["total_tokens"],
            )
        except Exception as e:
            logger.error(f"Zhipu API error: {e}")
            raise
    
    async def agenerate_response(self, prompt: str) -> LLMResult:
        """异步生成响应"""
        messages = self._construct_messages([prompt])
        while True:
            try:
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_key_list)
                api_client = ZhipuAPI(self.api_key_list[self.current_key_idx])
                
                # 使用真正的异步调用
                responses = []
                for msg in messages:
                    response = await api_client.async_chat_completion(msg, **self.args.dict())
                    responses.append(response)
                
                return responses
                
            except Exception as e:
                logger.info(f"Zhipu API error: {e}")
                logger.info("Retrying...")
                await asyncio.sleep(20)  # 使用异步sleep
                continue
    
    async def agenerate_response_without_construction(self, messages: str) -> LLMResult:
        """异步生成响应（不构造消息）"""
        while True:
            try:
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_key_list)
                api_client = ZhipuAPI(self.api_key_list[self.current_key_idx])
                
                # 使用真正的异步调用
                responses = []
                for msg in messages:
                    response = await api_client.async_chat_completion(msg, **self.args.dict())
                    responses.append(response)
                
                return responses
                
            except Exception as e:
                logger.info(f"Zhipu API error: {e}")
                logger.info("Retrying...")
                await asyncio.sleep(20)  # 使用异步sleep
                continue


@llm_registry.register("glm-4-completion")
class ZhipuCompletion(BaseCompletionModel):
    args: ZhipuCompletionArgs = Field(default_factory=ZhipuCompletionArgs)
    api_key_list: list = []
    current_key_idx = 0
    
    def __init__(self, max_retry: int = 15, **kwargs):
        args = ZhipuCompletionArgs()
        args = args.dict()
        
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        
        super().__init__(args=args, max_retry=max_retry)
        self.api_key_list = kwargs.pop('api_key_list', [])
        if not self.api_key_list:
            self.api_key_list = [ZHIPU_API_KEY] if ZHIPU_API_KEY else []
    
    def generate_response(self, prompt: str) -> LLMResult:
        """生成响应"""
        try:
            api_client = ZhipuAPI(self.api_key_list[self.current_key_idx])
            # 使用chat接口模拟completion
            response = api_client.chat_completion([{"role": "user", "content": prompt}], **self.args.dict())
            
            return LLMResult(
                content=response["choices"][0]["message"]["content"],
                send_tokens=response["usage"]["prompt_tokens"],
                recv_tokens=response["usage"]["completion_tokens"],
                total_tokens=response["usage"]["total_tokens"],
            )
        except Exception as e:
            logger.error(f"Zhipu API error: {e}")
            raise
    
    async def agenerate_response(self, prompt: str) -> LLMResult:
        """异步生成响应"""
        while True:
            try:
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_key_list)
                api_client = ZhipuAPI(self.api_key_list[self.current_key_idx])
                
                response = await api_client.async_chat_completion([{"role": "user", "content": prompt}], **self.args.dict())
                
                return [response["choices"][0]["message"]["content"]]
                
            except Exception as e:
                logger.info(f"Zhipu API error: {e}")
                logger.info("Retrying...")
                await asyncio.sleep(20)  # 使用异步sleep
                continue


@llm_registry.register("embedding-2")
class ZhipuEmbedding(BaseCompletionModel):
    args: ZhipuEmbeddingArgs = Field(default_factory=ZhipuEmbeddingArgs)
    api_key_list: list = []
    current_key_idx = 0
    
    def __init__(self, max_retry: int = 3, **kwargs):
        args = ZhipuEmbeddingArgs()
        args = args.dict()
        
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        
        super().__init__(args=args, max_retry=max_retry)
        self.api_key_list = kwargs.pop('api_key_list', [])
        if not self.api_key_list:
            self.api_key_list = [ZHIPU_API_KEY] if ZHIPU_API_KEY else []
    
    def generate_response(self, prompt: str) -> Any:
        """生成嵌入"""
        try:
            api_client = ZhipuAPI(self.api_key_list[self.current_key_idx])
            response = api_client.embedding(prompt, **self.args.dict())
            
            # 返回原始响应，保持与OpenAI兼容
            return response
        except Exception as e:
            logger.error(f"Zhipu API error: {e}")
            raise
    
    async def agenerate_response(self, sentences: Union[str, List[str]]) -> LLMResult:
        """异步生成嵌入"""
        if isinstance(sentences, str):
            sentences = [sentences]
        
        while True:
            try:
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_key_list)
                api_client = ZhipuAPI(self.api_key_list[self.current_key_idx])
                
                # 嵌入接口通常是同步的，使用run_in_executor
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: api_client.embedding(sentences, **self.args.dict())
                )
                
                return response
                
            except Exception as e:
                logger.info(f"Zhipu API error: {e}")
                logger.info("Retrying...")
                await asyncio.sleep(20)  # 使用异步sleep
                continue
