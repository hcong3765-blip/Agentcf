#!/usr/bin/env python3
"""
测试AgentCF修复后的代码
验证训练+测试和仅测试模式的结果是否一致
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(cmd, description):
    """运行命令并返回结果"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"📝 命令: {cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        print(f"⏱️  执行时间: {end_time - start_time:.2f} 秒")
        print(f"📤 返回码: {result.returncode}")
        
        if result.stdout:
            print(f"📋 标准输出:\n{result.stdout}")
        if result.stderr:
            print(f"⚠️  错误输出:\n{result.stderr}")
            
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print("⏰ 命令执行超时")
        return False, "", "Timeout"
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")
        return False, "", str(e)

def extract_ndcg_score(output):
    """从输出中提取NDCG@1分数"""
    lines = output.split('\n')
    for line in lines:
        if 'NDCG@1' in line or 'ndcg@1' in line:
            # 尝试提取数字
            import re
            numbers = re.findall(r'[\d.]+', line)
            if numbers:
                return float(numbers[0])
    return None

def main():
    """主测试函数"""
    print("🎯 AgentCF修复测试开始")
    print(f"🕒 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试参数
    model_name = "AgentCF"
    dataset_name = "CDs-100-user-dense"
    epochs = 1  # 使用1个epoch进行快速测试
    
    # 基础命令参数
    base_params = [
        f"--model {model_name}",
        f"--dataset {dataset_name}",
        f"--epochs {epochs}",
        "--train_batch_size 20",
        "--eval_batch_size 200",
        "--max_his_len 20",
        "--MAX_ITEM_LIST_LENGTH 20",
        "--shuffle False",
        "--api_batch 20"
    ]
    
    results = {}
    
    # 测试1: 训练+测试模式
    print(f"\n{'='*80}")
    print("🧪 测试1: 训练+测试模式")
    print(f"{'='*80}")
    
    train_test_cmd = f"cd agentcf && python run.py {' '.join(base_params)} --test_only=False"
    success, stdout, stderr = run_command(train_test_cmd, "训练+测试模式")
    
    if success:
        ndcg_score = extract_ndcg_score(stdout)
        results['train_test'] = ndcg_score
        print(f"✅ 训练+测试模式成功，NDCG@1: {ndcg_score}")
    else:
        print("❌ 训练+测试模式失败")
        results['train_test'] = None
    
    # 等待一下，确保文件保存完成
    time.sleep(2)
    
    # 测试2: 仅测试模式
    print(f"\n{'='*80}")
    print("🧪 测试2: 仅测试模式")
    print(f"{'='*80}")
    
    test_only_cmd = f"cd agentcf && python run.py {' '.join(base_params)} --test_only=True"
    success, stdout, stderr = run_command(test_only_cmd, "仅测试模式")
    
    if success:
        ndcg_score = extract_ndcg_score(stdout)
        results['test_only'] = ndcg_score
        print(f"✅ 仅测试模式成功，NDCG@1: {ndcg_score}")
    else:
        print("❌ 仅测试模式失败")
        results['test_only'] = None
    
    # 结果分析
    print(f"\n{'='*80}")
    print("📊 测试结果分析")
    print(f"{'='*80}")
    
    print(f"训练+测试模式 NDCG@1: {results['train_test']}")
    print(f"仅测试模式 NDCG@1: {results['test_only']}")
    
    if results['train_test'] is not None and results['test_only'] is not None:
        diff = abs(results['train_test'] - results['test_only'])
        print(f"差异: {diff:.4f}")
        
        if diff < 0.01:  # 差异小于0.01认为一致
            print("🎉 测试通过！两种模式结果一致")
            return True
        else:
            print("❌ 测试失败！两种模式结果不一致")
            return False
    else:
        print("❌ 测试失败！无法获取有效结果")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
