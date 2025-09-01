# 智谱GLM配置说明

## 1. 获取API密钥

1. 访问智谱AI官网：https://open.bigmodel.cn/
2. 注册账号并登录
3. 在控制台中创建API密钥
4. 复制API密钥

## 2. 设置环境变量

### Windows (PowerShell)
```powershell
$env:ZHIPU_API_KEY="your_api_key_here"
```

### Windows (CMD)
```cmd
set ZHIPU_API_KEY=your_api_key_here
```

### Linux/Mac
```bash
export ZHIPU_API_KEY="your_api_key_here"
```

## 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 4. 验证配置

运行以下命令验证配置是否正确：

```python
import os
from agentverse.llms.zhipu import ZhipuAPI

# 检查API密钥
api_key = os.environ.get("ZHIPU_API_KEY")
if api_key:
    print("API密钥已配置")
    
    # 测试API连接
    try:
        client = ZhipuAPI(api_key)
        print("智谱GLM API连接成功")
    except Exception as e:
        print(f"API连接失败: {e}")
else:
    print("请先设置ZHIPU_API_KEY环境变量")
```

## 5. 模型说明

- **glm-4**: 智谱GLM-4大模型，用于对话和文本生成
- **glm-3-turbo**: 智谱GLM-3-Turbo模型，更快的响应速度
- **embedding-2**: 智谱文本嵌入模型，用于生成文本向量

## 6. 注意事项

1. 确保网络环境能够访问智谱AI的API
2. API调用有频率限制，请合理使用
3. 建议在生产环境中使用环境变量文件管理密钥
4. 如遇到问题，请查看智谱AI官方文档：https://open.bigmodel.cn/doc/api
