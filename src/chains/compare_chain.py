"""
对比分析链 (Compare Chain)

对两款食品的配料表进行对比分析，推荐更健康的选择。
"""
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.chains.base import load_prompt_template

logger = logging.getLogger("foodguard.compare_chain")


_DEFAULT_COMPARE_PROMPT = """你是食品对比专家。请对比以下两款食品的配料表。

知识库信息：
{context}

食品A配料表：{food_a}
食品B配料表：{food_b}

请给出对比分析和推荐。
"""


def build_compare_chain(llm, vectorstore):
    """
    构建对比分析 Chain。

    参数:
        llm: LLM 实例
        vectorstore: 向量数据库实例

    返回:
        LangChain Runnable

    使用示例:
        chain = build_compare_chain(llm, vectorstore)
        result = chain.invoke({
            "food_a": "配料: 水、白砂糖、安赛蜜、柠檬酸...",
            "food_b": "配料: 水、赤藓糖醇、柠檬汁...",
        })
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 15},
    )

    prompt_template = load_prompt_template("compare.md", _DEFAULT_COMPARE_PROMPT)
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        if not docs:
            return "（知识库中暂无相关数据）"
        formatted = []
        for doc in docs:
            meta = doc.metadata
            formatted.append(
                f"【{meta.get('name', '')}】"
                f"功能:{meta.get('function', '')} | "
                f"风险:{meta.get('risk_level', '')} | "
                f"说明:{meta.get('risk_reason', '')}"
            )
        return "\n".join(formatted)

    # 对比分析需要同时检索两款食品的配料
    def retrieve_and_format(inputs: dict) -> str:
        """分别检索两款食品的配料，合并去重"""
        # 检索食品A的配料
        docs_a = retriever.invoke(inputs.get("food_a", ""))
        # 检索食品B的配料
        docs_b = retriever.invoke(inputs.get("food_b", ""))

        # 合并去重（按名称）
        seen = set()
        all_docs = []
        for doc in docs_a + docs_b:
            name = doc.metadata.get("name", "")
            if name not in seen:
                seen.add(name)
                all_docs.append(doc)

        return format_docs(all_docs)

    chain = (
        {
            "context": retrieve_and_format,
            "food_a": lambda x: x.get("food_a", ""),
            "food_b": lambda x: x.get("food_b", ""),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("对比分析链 (Compare Chain) 构建完成")
    return chain
