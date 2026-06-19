"""
ChromaDB 向量数据库封装

提供：
  - 向量存储的创建/加载
  - 相似度检索（用于 RAG 的检索步骤）
  - 带元数据过滤的检索
"""
import logging
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME, RETRIEVAL_TOP_K, logger


def get_vectorstore(
    embedding_function,
    collection_name: str = None,
    persist_directory: str | Path = None,
) -> Chroma:
    """
    获取或创建 Chroma 向量数据库实例。

    如果指定路径下已有持久化的向量库，直接加载；
    如果不存在，创建一个新的（后续通过 add_documents 填充）。

    参数:
        embedding_function: Embedding 函数实例（如 BGE-M3）
        collection_name: 集合名称，默认使用配置值
        persist_directory: 持久化目录，默认使用配置值

    返回:
        Chroma 实例

    使用示例:
        from src.models.embeddings import get_embeddings
        from src.data.vectorstore import get_vectorstore

        embeddings = get_embeddings()
        vectorstore = get_vectorstore(embeddings)
        # 如果已有数据：
        results = vectorstore.similarity_search("安赛蜜是什么", k=5)
    """
    if collection_name is None:
        collection_name = CHROMA_COLLECTION_NAME
    if persist_directory is None:
        persist_directory = str(CHROMA_DB_DIR)

    # 确保目录存在
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    logger.info(
        f"正在连接 ChromaDB: collection={collection_name}, path={persist_directory}"
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )

    logger.info(f"ChromaDB 连接完成")
    return vectorstore


def build_vectorstore_from_documents(
    documents: list[Document],
    embedding_function,
    collection_name: str = None,
    persist_directory: str | Path = None,
) -> Chroma:
    """
    从 Document 列表构建（或重建）向量数据库。

    会清空已有数据并重新写入。用于数据更新后的重建。

    参数:
        documents: LangChain Document 列表
        embedding_function: Embedding 函数实例
        collection_name: 集合名称
        persist_directory: 持久化目录

    返回:
        Chroma 实例
    """
    if collection_name is None:
        collection_name = CHROMA_COLLECTION_NAME
    if persist_directory is None:
        persist_directory = str(CHROMA_DB_DIR)

    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"正在构建向量数据库: {len(documents)} 个文档")

    # Chroma.from_documents 会把所有文档 Embed 后存入
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        collection_name=collection_name,
        persist_directory=str(persist_path),
    )

    logger.info(f"向量数据库构建完成，已持久化到: {persist_path}")
    return vectorstore


def retrieve(
    vectorstore: Chroma,
    query: str,
    top_k: int = None,
    filter_risk_level: Optional[str] = None,
) -> list[Document]:
    """
    从向量数据库中检索与 query 最相似的 top_k 个文档。

    参数:
        vectorstore: Chroma 实例
        query: 检索查询文本
        top_k: 返回条数，默认使用配置值
        filter_risk_level: 可选，按风险等级过滤（如只检索 'avoid' 级别的）

    返回:
        list[Document]，按相似度从高到低排列

    使用示例:
        results = retrieve(vectorstore, "防腐剂 安全吗", top_k=5)
        # 只检索高风险添加剂
        results = retrieve(vectorstore, "色素", filter_risk_level="avoid")
    """
    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    # 构造过滤条件
    search_kwargs = {"k": top_k}
    if filter_risk_level:
        search_kwargs["filter"] = {"risk_level": filter_risk_level}

    # 执行检索
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    results = retriever.invoke(query)
    logger.info(
        f"检索完成: query='{query[:50]}...', top_k={top_k}, "
        f"实际返回 {len(results)} 条"
    )
    return results
