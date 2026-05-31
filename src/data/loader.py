"""
自定义 Document Loader

从 JSON 知识库文件中加载添加剂数据，转换为 LangChain Document 对象。

每个添加剂条目 → 一个 Document：
  - page_content: 供 Embedding 和检索使用的文本
  - metadata: 结构化字段（name, cns, risk_level, function 等）
"""
import json
import logging
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document

logger = logging.getLogger("foodguard.loader")


class AdditiveJSONLoader:
    """
    从 JSON 知识库文件加载添加剂数据，生成 LangChain Document 列表。

    使用方式:
        loader = AdditiveJSONLoader("data/processed/additives_knowledge.json")
        docs = list(loader.lazy_load())  # 惰性加载，节省内存
    """

    def __init__(self, file_path: str | Path):
        """
        参数:
            file_path: JSON 知识库文件路径
        """
        self.file_path = Path(file_path)

    def lazy_load(self) -> Iterator[Document]:
        """
        惰性加载：一次生成一个 Document，适合大文件。

        每条添加剂被转换为一个 Document：
          - page_content: 用描述性的自然语言拼接，供 Embedding 和相似度检索
          - metadata: 保留结构化字段，供后续过滤和展示
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"知识库文件不存在: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"正在加载知识库: {self.file_path}，共 {len(data)} 条")

        for item in data:
            # ---- 构造 page_content ----
            # 这是 VectorStore 做相似度检索时的文本。
            # 原则：覆盖所有"用户可能用到的搜索词"。
            content_parts = [
                f"添加剂名称：{item.get('name', '')}",
                f"英文名称：{item.get('name_en', '')}",
                f"别名：{'、'.join(item.get('aliases', []))}",
                f"CNS编号：{item.get('cns', '')}",
                f"INS编号：{item.get('ins', '')}",
                f"功能类别：{item.get('function', '')}",
                f"风险等级：{item.get('risk_level', '')}",
                f"风险说明：{item.get('risk_reason', '')}",
                f"每日摄入限量：{item.get('daily_intake_limit', '')}",
                f"详细描述：{item.get('description', '')}",
            ]

            # 拼接使用范围
            usages = item.get("usages", [])
            if usages:
                usage_texts = []
                for u in usages:
                    usage_texts.append(
                        f"{u.get('food_name', '')}: 最大用量 {u.get('max_usage', '')}"
                    )
                content_parts.append(f"使用范围：{'；'.join(usage_texts)}")

            page_content = "\n".join(filter(None, content_parts))

            # ---- 构造 metadata ----
            # 保留结构化字段，后续 Chains 可以直接从 metadata 取值
            # 注意：ChromaDB 不允许 metadata 中有空 list，需要转换
            aliases = item.get("aliases", [])
            allergens = item.get("allergens", [])
            metadata = {
                "name": item.get("name", ""),
                "name_en": item.get("name_en", ""),
                "aliases": ", ".join(aliases) if aliases else "",
                "cns": item.get("cns", ""),
                "ins": item.get("ins", ""),
                "function": item.get("function", ""),
                "risk_level": item.get("risk_level", "safe"),
                "risk_reason": item.get("risk_reason", ""),
                "children_safe": item.get("children_safe", True),
                "pregnancy_safe": item.get("pregnancy_safe", True),
                "allergens": ", ".join(allergens) if allergens else "",
                "daily_intake_limit": item.get("daily_intake_limit", ""),
            }

            yield Document(page_content=page_content, metadata=metadata)

        logger.info(f"加载完成: {len(data)} 条添加剂数据")


def load_documents(file_path: str | Path = None) -> list[Document]:
    """
    便捷函数：一次性加载所有 Document。

    参数:
        file_path: JSON 知识库路径，默认使用 processed/additives_knowledge.json

    返回:
        list[Document]
    """
    from config import PROCESSED_DATA_DIR

    if file_path is None:
        file_path = PROCESSED_DATA_DIR / "additives_knowledge.json"

    loader = AdditiveJSONLoader(file_path)
    return list(loader.lazy_load())
