# ========== 导入模块部分 ==========
import importlib                                   # Python动态导入模块的核心库
import asyncio                                     # 异步编程支持
import os                                          # 操作系统相关功能
from agentverse.llms.zhipu import ZhipuAPI        # 智谱AI的API客户端

# ========== 智谱API配置 ==========
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")   # 从环境变量获取API密钥
if not ZHIPU_API_KEY:
    # 如果环境变量没有设置，使用硬编码的API密钥作为后备方案
    ZHIPU_API_KEY = "3bff6e47e2ef49b48cec36b3022332b5.zQTK7o7lr7jtpsrX"
    print("Using hardcoded API key as fallback")   # 🐛调试点：确认API密钥来源

# 创建智谱GLM API客户端实例
zhipu_client = ZhipuAPI(ZHIPU_API_KEY)

# ========== 其他导入 ==========
from recbole.utils import get_model as recbole_get_model  # RecBole的模型获取函数
from collections import defaultdict               # 默认字典，自动创建不存在的键
import os, torch, random, numpy as np, pandas as pd, json, pickle, gzip  # 各种常用库
from tqdm import tqdm                             # 进度条显示

# ========== 布尔值定义 ==========
true = True                                       # 兼容性设置
false = False

def check_path(path):
    """
    检查路径是否存在，如果不存在则创建
    参数：path - 要检查的路径
    """
    if not os.path.exists(path):                  # 如果路径不存在
        os.makedirs(path)                         # 递归创建目录

def get_model(model_name):
    """
    🔥 核心函数：动态获取模型类
    这是整个项目中最重要的函数之一，实现了模型的动态加载
    参数：model_name - 模型名称字符串（如'AgentCF'）
    返回：模型类对象
    """
    # 第一步：检查自定义模型是否存在
    if importlib.util.find_spec(f'model.{model_name.lower()}', __name__):
        # 💡语法解释：importlib.util.find_spec()检查模块是否存在，不实际导入
        # f'model.{model_name.lower()}'构造模块路径，如'model.agentcf'
        
        # 第二步：动态导入模块
        model_module = importlib.import_module(f'model.{model_name.lower()}', __name__)
        # 💡语法解释：importlib.import_module()动态导入模块，相当于import model.agentcf
        
        # 第三步：从模块中获取类
        model_class = getattr(model_module, model_name)
        # 💡语法解释：getattr(对象, 属性名)获取对象的属性，这里获取AgentCF类
        
        return model_class
    else:
        # 如果自定义模型不存在，使用RecBole的默认模型加载方式
        return recbole_get_model(model_name)

async def dispatch_zhipu_requests(messages_list, model: str, temperature: float):
    """
    异步分发智谱GLM API请求
    💡语法解释：async def定义异步函数，可以使用await关键字
    参数：
    - messages_list: 要发送的消息列表
    - model: 使用的模型名称
    - temperature: 温度参数，控制生成的随机性
    返回：API响应列表
    """
    async_responses = []                           # 存储异步响应的列表
    
    for x in messages_list:                       # 遍历每个消息
        # 使用线程池执行器运行同步函数
        response = await asyncio.get_event_loop().run_in_executor(
            None,                                 # 使用默认执行器
            lambda: zhipu_client.chat_completion( # lambda匿名函数
                messages=x,
                model=model,
                temperature=temperature
            )
        )
        async_responses.append(response)
    
    return async_responses

def dispatch_single_zhipu_requests(message, model: str, temperature: float):
    """
    同步发送单个智谱GLM API请求
    参数说明同上，但这里是同步函数
    """
    response = zhipu_client.chat_completion(
        messages=message,
        model=model,
        temperature=temperature
    )
    return response

# ========== Amazon数据集名称映射 ==========
amazon_dataset2fullname = {
    'Beauty': 'All_Beauty',
    'Fashion': 'AMAZON_FASHION',
    # ... 其他映射关系
    'Games': 'Video_Games'
}
# 💡作用：将简短的数据集名称映射到完整的Amazon产品类别名称