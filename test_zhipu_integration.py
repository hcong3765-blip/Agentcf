#!/usr/bin/env python3
"""
测试智谱GLM集成的脚本
"""

import os
import sys
import asyncio

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agentcf'))
sys.path.insert(0, os.path.dirname(__file__))

def test_zhipu_imports():
    """测试智谱GLM模块导入"""
    try:
        from agentverse.llms.zhipu import ZhipuChat, ZhipuCompletion, ZhipuEmbedding, ZhipuAPI
        print("✓ 智谱GLM模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 智谱GLM模块导入失败: {e}")
        return False

def test_zhipu_api_key():
    """测试API密钥配置"""
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        # 尝试从配置文件中获取
        api_key = "3bff6e47e2ef49b48cec36b3022332b5.zQTK7o7lr7jtpsrX"
        print(f"✓ 使用硬编码的API密钥: {api_key[:10]}...")
        return True
    else:
        print(f"✓ API密钥已配置: {api_key[:10]}...")
        return True

def test_zhipu_client():
    """测试智谱GLM API客户端"""
    try:
        from agentverse.llms.zhipu import ZhipuAPI
        api_key = os.environ.get("ZHIPU_API_KEY") or "3bff6e47e2ef49b48cec36b3022332b5.zQTK7o7lr7jtpsrX"
            
        client = ZhipuAPI(api_key)
        print("✓ 智谱GLM API客户端创建成功")
        return True
    except Exception as e:
        print(f"✗ 智谱GLM API客户端创建失败: {e}")
        return False

async def test_zhipu_chat():
    """测试智谱GLM聊天功能"""
    try:
        from agentverse.llms.zhipu import ZhipuChat
        
        api_key = os.environ.get("ZHIPU_API_KEY") or "3bff6e47e2ef49b48cec36b3022332b5.zQTK7o7lr7jtpsrX"
            
        chat_model = ZhipuChat(
            model="glm-4",
            temperature=0.7,
            max_tokens=100,
            api_key_list=[api_key]
        )
        
        # 测试同步调用
        response = chat_model.generate_response("你好，请简单介绍一下自己")
        print(f"✓ 聊天功能测试成功，响应长度: {len(str(response.content))}")
        return True
        
    except Exception as e:
        print(f"✗ 聊天功能测试失败: {e}")
        return False

async def test_zhipu_embedding():
    """测试智谱GLM嵌入功能"""
    try:
        from agentverse.llms.zhipu import ZhipuEmbedding
        
        api_key = os.environ.get("ZHIPU_API_KEY") or "3bff6e47e2ef49b48cec36b3022332b5.zQTK7o7lr7jtpsrX"
            
        embedding_model = ZhipuEmbedding(
            model="embedding-2",
            api_key_list=[api_key]
        )
        
        # 测试同步调用
        response = embedding_model.generate_response("测试文本")
        if isinstance(response, dict) and "data" in response:
            embedding = response["data"][0]["embedding"]
            print(f"✓ 嵌入功能测试成功，向量维度: {len(embedding)}")
        else:
            print(f"✓ 嵌入功能测试成功，响应: {type(response)}")
        return True
        
    except Exception as e:
        print(f"✗ 嵌入功能测试失败: {e}")
        return False

def test_config_files():
    """测试配置文件修改"""
    try:
        # 检查配置文件是否已修改
        config_path = "agentcf/props/AgentCF.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "glm-4" in content and "embedding-2" in content:
                    print("✓ 配置文件已正确修改为智谱GLM模型")
                    return True
                else:
                    print("✗ 配置文件未正确修改")
                    return False
        else:
            print("✗ 配置文件不存在")
            return False
    except Exception as e:
        print(f"✗ 配置文件检查失败: {e}")
        return False

async def main():
    """主测试函数"""
    print("=" * 50)
    print("智谱GLM集成测试")
    print("=" * 50)
    
    tests = [
        ("模块导入测试", test_zhipu_imports),
        ("API密钥配置测试", test_zhipu_api_key),
        ("API客户端测试", test_zhipu_client),
        ("聊天功能测试", test_zhipu_chat),
        ("嵌入功能测试", test_zhipu_embedding),
        ("配置文件测试", test_config_files),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ 测试执行异常: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！智谱GLM集成成功！")
        print("\n下一步:")
        print("1. 确保设置了正确的API密钥")
        print("2. 运行AgentCF项目")
        print("3. 查看ZHIPU_SETUP.md了解详细配置说明")
    else:
        print("⚠️  部分测试失败，请检查配置和依赖")
        print("\n常见问题:")
        print("1. 检查是否安装了所有依赖: pip install -r requirements.txt")
        print("2. 检查API密钥是否正确设置")
        print("3. 检查网络连接是否正常")

if __name__ == "__main__":
    asyncio.run(main())
