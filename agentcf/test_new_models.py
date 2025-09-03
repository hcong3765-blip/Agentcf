#!/usr/bin/env python3
"""
测试新模型 GLM-4.5 和 embedding-3 的配置
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_registration():
    """测试模型注册"""
    print("🔍 测试模型注册...")
    
    try:
        from agentverse.llms import llm_registry
        
        # 检查 GLM-4.5 注册
        if "glm-4.5" in llm_registry.get_all_llm_names():
            print("✅ GLM-4.5 模型已成功注册")
        else:
            print("❌ GLM-4.5 模型注册失败")
            
        # 检查 embedding-3 注册
        if "embedding-3" in llm_registry.get_all_llm_names():
            print("✅ embedding-3 模型已成功注册")
        else:
            print("❌ embedding-3 模型注册失败")
            
        print("📋 所有已注册的模型:")
        for model_name in sorted(llm_registry.get_all_llm_names()):
            print(f"   - {model_name}")
            
    except Exception as e:
        print(f"❌ 模型注册测试失败: {str(e)}")

def test_config_loading():
    """测试配置文件加载"""
    print("\n🔍 测试配置文件...")
    
    try:
        import yaml
        with open('props/AgentCF.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        print(f"📝 配置的 LLM 模型: {config.get('llm_model')}")
        print(f"📝 配置的 Embedding 模型: {config.get('embedding_model')}")
        
        if config.get('llm_model') == 'glm-4.5':
            print("✅ LLM 模型配置正确")
        else:
            print("❌ LLM 模型配置错误")
            
        if config.get('embedding_model') == 'embedding-3':
            print("✅ Embedding 模型配置正确")
        else:
            print("❌ Embedding 模型配置错误")
            
    except Exception as e:
        print(f"❌ 配置文件测试失败: {str(e)}")

def test_model_instantiation():
    """测试模型实例化"""
    print("\n🔍 测试模型实例化...")
    
    try:
        from agentverse.llms import llm_registry
        
        # 测试 GLM-4.5
        try:
            glm45_class = llm_registry.get_llm("glm-4.5")
            glm45_instance = glm45_class()
            print("✅ GLM-4.5 模型实例化成功")
            print(f"   默认模型: {glm45_instance.args.model}")
        except Exception as e:
            print(f"❌ GLM-4.5 模型实例化失败: {str(e)}")
        
        # 测试 embedding-3
        try:
            emb3_class = llm_registry.get_llm("embedding-3")
            emb3_instance = emb3_class()
            print("✅ embedding-3 模型实例化成功")
            print(f"   默认模型: {emb3_instance.args.model}")
        except Exception as e:
            print(f"❌ embedding-3 模型实例化失败: {str(e)}")
            
    except Exception as e:
        print(f"❌ 模型实例化测试失败: {str(e)}")

def main():
    """主函数"""
    print("🚀 开始测试新模型配置...\n")
    
    test_model_registration()
    test_config_loading()
    test_model_instantiation()
    
    print("\n🎉 测试完成!")

if __name__ == "__main__":
    main()

