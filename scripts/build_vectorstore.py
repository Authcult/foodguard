"""
构建向量数据库

从处理好的 JSON 知识库加载数据 → Document → Embedding → ChromaDB 持久化。

运行方式：
  python scripts/build_vectorstore.py

首次运行会：
  1. 下载 BGE-M3 模型（约 2.2GB，仅首次）
  2. 为每条添加剂生成 Embedding 向量
  3. 存入 ChromaDB（持久化到 data/chroma_db/）

后续运行会重建向量库（覆盖旧数据）。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_DIR, CHROMA_DB_DIR, logger


def main():
    """主流程"""
    logger.info("=" * 60)
    logger.info("开始构建向量数据库")
    logger.info("=" * 60)

    # 1. 检查知识库文件是否存在
    knowledge_path = PROCESSED_DATA_DIR / "additives_knowledge.json"
    if not knowledge_path.exists():
        logger.error(f"知识库文件不存在: {knowledge_path}")
        logger.error("请先运行: python scripts/process_data.py")
        sys.exit(1)

    # 2. 加载 Embedding 模型
    from src.models.embeddings import get_embeddings
    embeddings = get_embeddings()

    # 3. 加载 Document
    from src.data.loader import load_documents
    documents = load_documents(knowledge_path)
    logger.info(f"已加载 {len(documents)} 个 Document")

    # 4. 构建向量库
    from src.data.vectorstore import build_vectorstore_from_documents
    vectorstore = build_vectorstore_from_documents(
        documents=documents,
        embedding_function=embeddings,
    )

    logger.info("=" * 60)
    logger.info("向量数据库构建完成！")
    logger.info(f"持久化目录: {CHROMA_DB_DIR}")
    logger.info("=" * 60)

    # 5. 简单测试
    logger.info("正在测试检索...")
    results = vectorstore.similarity_search("防腐剂 安全吗", k=3)
    logger.info(f"检索测试结果 ({len(results)} 条):")
    for i, doc in enumerate(results, 1):
        logger.info(f"  {i}. {doc.metadata['name']} [{doc.metadata['risk_level']}]")


if __name__ == "__main__":
    main()
