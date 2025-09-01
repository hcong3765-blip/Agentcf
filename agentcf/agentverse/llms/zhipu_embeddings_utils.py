"""
智谱GLM嵌入工具模块
替代OpenAI的embeddings_utils
"""

import numpy as np
from typing import List, Union, Tuple
import os
from .zhipu import ZhipuAPI

# 获取API密钥
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")
if not ZHIPU_API_KEY:
    # 使用硬编码的API密钥作为后备
    ZHIPU_API_KEY = "3bff6e47e2ef49b48cec36b3022332b5.zQTK7o7lr7jtpsrX"
    print("Using hardcoded API key as fallback in zhipu_embeddings_utils")

# 创建API客户端
zhipu_client = ZhipuAPI(ZHIPU_API_KEY)


def get_embedding(text: str, model: str = "embedding-2") -> List[float]:
    """
    获取单个文本的嵌入向量
    
    Args:
        text: 输入文本
        model: 嵌入模型名称
    
    Returns:
        嵌入向量列表
    """
    try:
        response = zhipu_client.embedding(text, model=model)
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []


def get_embeddings(texts: List[str], model: str = "embedding-2") -> List[List[float]]:
    """
    获取多个文本的嵌入向量
    
    Args:
        texts: 输入文本列表
        model: 嵌入模型名称
    
    Returns:
        嵌入向量列表的列表
    """
    try:
        response = zhipu_client.embedding(texts, model=model)
        return [item["embedding"] for item in response["data"]]
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return []


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine"
) -> List[float]:
    """
    计算查询嵌入向量与所有嵌入向量的距离
    
    Args:
        query_embedding: 查询嵌入向量
        query_embedding: 所有嵌入向量
        distance_metric: 距离度量方法 ("cosine", "euclidean", "manhattan")
    
    Returns:
        距离列表
    """
    query_embedding = np.array(query_embedding)
    embeddings = np.array(embeddings)
    
    if distance_metric == "cosine":
        # 余弦距离
        query_norm = np.linalg.norm(query_embedding)
        embeddings_norm = np.linalg.norm(embeddings, axis=1)
        
        # 避免除零错误
        if query_norm == 0:
            return [1.0] * len(embeddings)
        
        embeddings_norm = np.where(embeddings_norm == 0, 1e-8, embeddings_norm)
        
        similarities = np.dot(embeddings, query_embedding) / (embeddings_norm * query_norm)
        distances = 1 - similarities
        
    elif distance_metric == "euclidean":
        # 欧几里得距离
        distances = np.linalg.norm(embeddings - query_embedding, axis=1)
        
    elif distance_metric == "manhattan":
        # 曼哈顿距离
        distances = np.sum(np.abs(embeddings - query_embedding), axis=1)
        
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    
    return distances.tolist()


def indices_of_nearest_neighbors_from_distances(
    distances: List[float], k: int = 1
) -> List[int]:
    """
    根据距离找到最近的k个邻居的索引
    
    Args:
        distances: 距离列表
        k: 返回的最近邻居数量
    
    Returns:
        最近邻居的索引列表
    """
    distances = np.array(distances)
    indices = np.argsort(distances)[:k]
    return indices.tolist()


def tsne_components_from_embeddings(
    embeddings: List[List[float]], 
    n_components: int = 2,
    random_state: int = 42
) -> List[List[float]]:
    """
    使用t-SNE将高维嵌入向量降维到指定维度
    
    Args:
        embeddings: 嵌入向量列表
        n_components: 目标维度
        random_state: 随机种子
    
    Returns:
        降维后的向量列表
    """
    try:
        from sklearn.manifold import TSNE
        
        tsne = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=min(30, len(embeddings) - 1)
        )
        
        embeddings_array = np.array(embeddings)
        components = tsne.fit_transform(embeddings_array)
        
        return components.tolist()
        
    except ImportError:
        print("scikit-learn is required for t-SNE. Install with: pip install scikit-learn")
        return embeddings


def chart_from_components(
    components: List[List[float]],
    labels: List[str] = None,
    width: int = 800,
    height: int = 600
) -> str:
    """
    从降维后的组件创建图表（返回HTML字符串）
    
    Args:
        components: 降维后的向量列表
        labels: 标签列表
        width: 图表宽度
        height: 图表高度
    
    Returns:
        HTML图表字符串
    """
    try:
        import plotly.graph_objects as go
        
        x_coords = [comp[0] for comp in components]
        y_coords = [comp[1] for comp in components]
        
        if labels is None:
            labels = [f"Point {i}" for i in range(len(components))]
        
        fig = go.Figure(data=go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            text=labels,
            textposition="top center",
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="t-SNE Visualization of Embeddings",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            width=width,
            height=height
        )
        
        return fig.to_html(include_plotlyjs='cdn')
        
    except ImportError:
        print("plotly is required for chart generation. Install with: pip install plotly")
        return f"Chart generation requires plotly. Components: {components[:5]}..."


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    计算两个向量的余弦相似度
    
    Args:
        a: 第一个向量
        b: 第二个向量
    
    Returns:
        余弦相似度值
    """
    a = np.array(a)
    b = np.array(b)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """
    计算两个向量的欧几里得距离
    
    Args:
        a: 第一个向量
        b: 第二个向量
    
    Returns:
        欧几里得距离值
    """
    a = np.array(a)
    b = np.array(b)
    
    return np.linalg.norm(a - b)


def manhattan_distance(a: List[float], b: List[float]) -> float:
    """
    计算两个向量的曼哈顿距离
    
    Args:
        a: 第一个向量
        b: 第二个向量
    
    Returns:
        曼哈顿距离值
    """
    a = np.array(a)
    b = np.array(b)
    
    return np.sum(np.abs(a - b))
