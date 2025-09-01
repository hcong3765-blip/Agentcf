from agentverse.registry import Registry

llm_registry = Registry(name="LLMRegistry")

from .base import BaseLLM, BaseChatModel, BaseCompletionModel, LLMResult
from .openai import OpenAIChat, OpenAICompletion, OpenAIEmbedding
from .zhipu import ZhipuChat, ZhipuCompletion, ZhipuEmbedding
