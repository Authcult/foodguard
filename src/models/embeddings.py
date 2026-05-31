"""
Embedding 模型初始化

使用 BGE-M3（BAAI/bge-m3），通过 langchain-community 的 HuggingFaceEmbeddings 加载。

BGE-M3 特点：
  - 支持中文，多语言能力好
  - 支持密集(Dense)和稀疏(Sparse)两种检索方式
  - 最大输入长度 8192 tokens
  - 输出向量维度 1024
  - 在 MTEB 中文榜单上表现优秀

下载源（按优先级）：
  1. ModelScope（modelscope.cn，国内可访问）
  2. HuggingFace 镜像（hf-mirror.com）
  3. HuggingFace 官方（huggingface.co，需科学上网）

首次运行会自动下载模型（约 2.2GB），之后使用缓存。
"""
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, logger


# 全局单例
_embedding_model: HuggingFaceEmbeddings = None


def _download_from_modelscope(model_name: str, cache_dir: str) -> str:
    """
    从 ModelScope 下载模型到本地，返回本地路径。

    ModelScope 是阿里维护的模型库，国内下载速度快。
    BGE-M3 在 ModelScope 上的路径同样是 BAAI/bge-m3。
    """
    try:
        from modelscope import snapshot_download
        logger.info("正在从 ModelScope (modelscope.cn) 下载模型...")
        local_path = snapshot_download(
            model_name,
            cache_dir=cache_dir,
        )
        logger.info(f"ModelScope 下载完成: {local_path}")
        return local_path
    except ImportError:
        logger.warning("modelscope 未安装，请运行: pip install modelscope")
        raise
    except Exception as e:
        logger.warning(f"ModelScope 下载失败: {e}")
        raise


def _get_model_path() -> str:
    """
    获取模型本地路径。按优先级尝试不同下载源。

    返回:
        模型本地目录路径
    """
    model_name = EMBEDDING_MODEL_NAME
    # 模型缓存目录
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
    cache_dir = os.path.abspath(cache_dir)

    # 检查是否已经下载过
    model_local_dir = os.path.join(cache_dir, model_name.replace("/", "_"))
    if os.path.exists(model_local_dir) and os.listdir(model_local_dir):
        logger.info(f"模型已缓存，直接使用: {model_local_dir}")
        return model_local_dir

    # 方案1：ModelScope（国内最可靠）
    try:
        return _download_from_modelscope(model_name, cache_dir)
    except Exception:
        pass

    # 方案2：HuggingFace 镜像（需要网络能访问 hf-mirror.com）
    hf_mirror = os.getenv("HF_ENDPOINT", "")
    if hf_mirror:
        logger.info(f"使用 HuggingFace 镜像: {hf_mirror}")
    else:
        # 自动尝试 hf-mirror
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("自动设置 HF_ENDPOINT=https://hf-mirror.com（国内镜像）")

    # 直接使用 HuggingFaceEmbeddings 自动下载
    # （HuggingFaceEmbeddings 内部会调用 huggingface_hub 下载）
    return model_name


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    获取 Embedding 模型实例（单例模式）。

    首次调用会下载模型（约 2.2GB），后续直接使用缓存。

    下载源自动选择（ModelScope → HF镜像 → HF官方）。

    返回:
        HuggingFaceEmbeddings 实例。

    使用示例:
        from src.models.embeddings import get_embeddings
        embeddings = get_embeddings()
        vector = embeddings.embed_query("安赛蜜是什么")
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    logger.info(f"正在加载 Embedding 模型: {EMBEDDING_MODEL_NAME}")
    logger.info("首次加载会下载模型文件（约 2.2GB），请耐心等待...")

    # 获取模型路径（自动选择下载源）
    model_path = _get_model_path()

    # encode_kwargs 用于控制模型行为
    encode_kwargs = {
        "normalize_embeddings": True,  # 向量归一化，便于余弦相似度计算
    }

    # model_kwargs 用于控制模型加载
    model_kwargs = {
        "device": EMBEDDING_DEVICE,
    }

    _embedding_model = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    logger.info(f"Embedding 模型加载完成: {EMBEDDING_MODEL_NAME}")
    return _embedding_model
