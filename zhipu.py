from zai import ZhipuAiClient
from typing import List, Dict, Optional, Any
import os

class ZhipuAPIClient:
    def __init__(self, api_key: Optional[str] = None):
        """初始化智谱AI客户端
        
        Args:
            api_key: 智谱API密钥，若未提供则尝试从环境变量ZHIPU_API_KEY获取
        """
        api_key = api_key or os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("API密钥未提供，请传入api_key参数或设置ZHIPU_API_KEY环境变量")
            
        self.client = ZhipuAiClient(api_key=api_key)
        
    def embedding(self, texts: List[str], model: str = "embedding-2") -> Dict[str, Any]:
        """
        文本嵌入功能
        
        :param texts: 要嵌入的文本列表
        :param model: 嵌入模型，默认为embedding-2
        :return: 包含嵌入结果的字典
        """
        if not texts or not all(isinstance(t, str) for t in texts):
            return {
                "success": False,
                "data": None,
                "message": "输入必须是非空的字符串列表"
            }
            
        try:
            response = self.client.embeddings.create(
                model=model,
                input=texts
            )
            # 提取嵌入向量
            embeddings = [item.embedding for item in response.data]
            return {
                "success": True,
                "data": embeddings,
                "message": "嵌入成功"
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "message": f"嵌入失败: {str(e)}"
            }
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             model: str = "glm-4.5", 
             temperature: float = 0.6, 
             max_tokens: int = 4096, 
             thinking_enabled: bool = False) -> Dict[str, Any]:
        """
        对话功能
        
        :param messages: 消息列表，格式为[{"role": "user/assistant", "content": "..."}]
        :param model: 对话模型，默认为glm-4.5
        :param temperature: 温度参数，控制输出随机性(0-1)
        :param max_tokens: 最大输出token数
        :param thinking_enabled: 是否启用深度思考模式
        :return: 包含对话回复的字典
        """
        if not messages or not all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
            return {
                "success": False,
                "data": None,
                "message": "消息格式不正确，应为包含role和content的字典列表"
            }
            
        try:
            # 构建参数
            params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # 仅在模型支持时添加thinking参数
            if thinking_enabled and model in ["glm-4", "glm-4.5"]:
                params["thinking"] = {"type": "enabled"}
            
            response = self.client.chat.completions.create(**params)
            
            return {
                "success": True,
                "data": {
                    "content": response.choices[0].message.content,
                    "role": response.choices[0].message.role,
                    "usage": response.usage.model_dump() if hasattr(response, 'usage') else None
                },
                "message": "对话成功"
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "message": f"对话失败: {str(e)}"
            }
    
    def completion(self, 
                  prompt: str, 
                  model: str = "glm-4.5", 
                  temperature: float = 0.6, 
                  max_tokens: int = 4096) -> Dict[str, Any]:
        """
        文本补全功能
        
        :param prompt: 提示文本
        :param model: 补全模型，默认为glm-4.5
        :param temperature: 温度参数，控制输出随机性
        :param max_tokens: 最大输出token数
        :return: 包含补全结果的字典
        """
        if not isinstance(prompt, str) or not prompt.strip():
            return {
                "success": False,
                "data": None,
                "message": "提示文本必须是非空字符串"
            }
            
        try:
            # 对于补全功能，使用chat接口模拟
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return {
                "success": True,
                "data": {
                    "content": response.choices[0].message.content,
                    "usage": response.usage.model_dump() if hasattr(response, 'usage') else None
                },
                "message": "补全成功"
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "message": f"补全失败: {str(e)}"
            }


# 测试代码
def test_zhipu_api():
    # 使用提供的API密钥
    API_KEY = "3bff6e47e2ef49b48cec36b3022332b5.zQTK7o7lr7jtpsrX"
    
    if not API_KEY:
        print("请先设置ZHIPU_API_KEY环境变量或直接传入API密钥")
        return
    
    try:
        client = ZhipuAPIClient(API_KEY)
        
        print("=== 测试Embedding功能 ===")
        texts = ["智谱AI开放平台", "人工智能技术", "自然语言处理"]
        embedding_result = client.embedding(texts)
        if embedding_result["success"]:
            print(f"嵌入数量: {len(embedding_result['data'])}")
            print(f"第一个嵌入向量长度: {len(embedding_result['data'][0])}")
        else:
            print(f"嵌入测试失败: {embedding_result['message']}")
        
        print("\n=== 测试Chat功能 ===")
        messages = [
            {"role": "user", "content": "什么是智谱AI开放平台？"},
        ]
        chat_result = client.chat(messages, thinking_enabled=True)
        if chat_result["success"]:
            print(f"角色: {chat_result['data']['role']}")
            print(f"回复: {chat_result['data']['content']}")
            print(f"用量: {chat_result['data']['usage']}")
        else:
            print(f"对话测试失败: {chat_result['message']}")
        
        # 多轮对话测试
        print("\n=== 测试多轮Chat功能 ===")
        if chat_result["success"]:
            messages.append({
                "role": chat_result["data"]["role"], 
                "content": chat_result["data"]["content"]
            })
            messages.append({"role": "user", "content": "它有哪些主要功能？"})
            
            chat_result2 = client.chat(messages)
            if chat_result2["success"]:
                print(f"角色: {chat_result2['data']['role']}")
                print(f"回复: {chat_result2['data']['content']}")
            else:
                print(f"多轮对话测试失败: {chat_result2['message']}")
        
        print("\n=== 测试Completion功能 ===")
        prompt = "请简要介绍智谱AI开放平台的优势，不超过200字。"
        completion_result = client.completion(prompt, temperature=0.8)
        if completion_result["success"]:
            print(f"补全结果: {completion_result['data']['content']}")
        else:
            print(f"补全测试失败: {completion_result['message']}")
            
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")


if __name__ == "__main__":
    test_zhipu_api()
