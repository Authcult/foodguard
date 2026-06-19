"""
Embedding 模型初始化

支持两种后端：
  1. Ollama（本地部署，默认）— 使用 bge-m3:latest
  2. HuggingFace（备用）— 使用 sentence-transformers 加载 BGE-M3

通过 config.EMBEDDING_BACKEND 切换。
"""
from config import (
    EMBEDDING_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    logger,
)

# 全局单例
_embedding_model = None


def get_embeddings():
    """
    获取 Embedding 模型实例（单例模式）。

    根据 config.EMBEDDING_BACKEND 自动选择 Ollama 或 HuggingFace。

    返回:
        Embedding 模型实例（支持 .embed_query() 和 .embed_documents()）
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    if EMBEDDING_BACKEND == "ollama":
        _embedding_model = _get_ollama_embeddings()
    elif EMBEDDING_BACKEND == "huggingface":
        _embedding_model = _get_huggingface_embeddings()
    else:
        raise ValueError(f"不支持的 Embedding 后端: {EMBEDDING_BACKEND}，可选: ollama, huggingface")

    return _embedding_model


def _get_ollama_embeddings():
    """初始化 Ollama Embedding"""
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    logger.info(f"Embedding 初始化完成 (Ollama): model={OLLAMA_EMBEDDING_MODEL}")
    return embeddings


def _get_huggingface_embeddings():
    """初始化 HuggingFace Embedding（备用，需要下载模型）"""
    import os
    from langchain_community.embeddings import HuggingFaceEmbeddings

    logger.info(f"正在加载 Embedding 模型 (HuggingFace): {EMBEDDING_MODEL_NAME}")

    # 尝试从 ModelScope 下载
    model_path = _try_download_model()

    encode_kwargs = {"normalize_embeddings": True}
    model_kwargs = {"device": EMBEDDING_DEVICE}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    logger.info(f"Embedding 初始化完成 (HuggingFace): {EMBEDDING_MODEL_NAME}")
    return embeddings


def _try_download_model() -> str:
    """尝试从 ModelScope 下载模型，失败则返回模型名让 HuggingFace 自行下载"""
    import os

    model_name = EMBEDDING_MODEL_NAME
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
    cache_dir = os.path.abspath(cache_dir)

    # 检查缓存
    model_local_dir = os.path.join(cache_dir, model_name.replace("/", "_"))
    if os.path.exists(model_local_dir) and os.listdir(model_local_dir):
        logger.info(f"模型已缓存: {model_local_dir}")
        return model_local_dir

    # 尝试 ModelScope
    try:
        from modelscope import snapshot_download
        logger.info("正在从 ModelScope 下载模型...")
        local_path = snapshot_download(model_name, cache_dir=cache_dir)
        logger.info(f"ModelScope 下载完成: {local_path}")
        return local_path
    except Exception as e:
        logger.warning(f"ModelScope 下载失败: {e}")

    # 尝试 HuggingFace 镜像
    if not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("自动设置 HF_ENDPOINT=https://hf-mirror.com")

    return model_name
