"""
文本分割器

每个配料条目已经是一个独立的 Document（由 loader 保证），不需要进一步切割。
但如果条目内容太长，可以使用此模块按语义分割。

当前实现：
  - IdentitySplitter: 不做任何分割，直接返回原 Document（默认使用）
  - 保留扩展空间，未来可以按添加剂描述、使用范围等分块
"""
import logging
from typing import Iterator

from langchain_core.documents import Document

logger = logging.getLogger("foodguard.splitter")


class IdentitySplitter:
    """
    不做分割的"分割器"，保持每个添加剂条目完整。

    原因：我们的数据粒度已经是"每条添加剂一个 Document"，嵌入后的检索
          也是按添加剂条目级别进行的，不需要进一步切分。
    """

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """直接返回原文档列表，不做任何修改"""
        logger.info(f"IdentitySplitter: 保持 {len(documents)} 个 Document 不变")
        return documents

    # lazy 版本，处理大列表时用
    def lazy_split_documents(self, documents: list[Document]) -> Iterator[Document]:
        for doc in documents:
            yield doc


def get_splitter():
    """
    获取文本分割器实例。

    返回:
        IdentitySplitter 实例
    """
    return IdentitySplitter()
