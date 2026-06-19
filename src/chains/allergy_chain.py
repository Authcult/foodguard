"""
过敏原检测链 (Allergy Check Chain)

根据用户已知过敏史，检测食品配料表中的过敏风险。
"""
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.chains.base import load_prompt_template, format_aliases, format_allergens

logger = logging.getLogger("foodguard.allergy_chain")


_DEFAULT_ALLERGY_PROMPT = """你是一位过敏医学专家。根据用户的过敏史和食品配料表，检测过敏风险。

用户过敏史：{user_allergens}
知识库信息：{context}
配料表：{question}

请检测并列出所有过敏风险。
"""


def build_allergy_chain(llm, vectorstore):
    """
    构建过敏原检测 Chain。

    参数:
        llm: LLM 实例
        vectorstore: 向量数据库实例

    返回:
        LangChain Runnable

    使用示例:
        chain = build_allergy_chain(llm, vectorstore)
        result = chain.invoke({
            "question": "配料: 大豆蛋白、酪蛋白酸钠、花生油...",
            "user_allergens": "花生、牛奶",
        })
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    prompt_template = load_prompt_template("allergy_check.md", _DEFAULT_ALLERGY_PROMPT)
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
                f"别名:{format_aliases(meta)} | "
                f"过敏原:{format_allergens(meta)} | "
                f"风险:{meta.get('risk_level', '')}"
            )
        return "\n".join(formatted)

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "user_allergens": lambda x: x.get("user_allergens", "未提供"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("过敏原检测链 (Allergy Chain) 构建完成")
    return chain
