"""
风险标注链 (Risk Label Chain)

基于检索到的知识库，对配料表中的每个配料标注风险等级。

与 interpret_chain 的区别：
  - interpret_chain: 自然语言解读，适合消费者理解
  - risk_chain: 结构化风险标注，每条配料对应一个风险标签

这个 Chain 展示了如何用 Pydantic 约束 LLM 的输出格式。
"""
import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import PROMPTS_DIR
from src.models.schemas import IngredientResult

logger = logging.getLogger("foodguard.risk_chain")


def _load_prompt_template(filename: str) -> str:
    """从 prompts 目录加载 Prompt 模板"""
    prompt_path = PROMPTS_DIR / filename
    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    return """你是一位食品安全风险评估专家。

根据以下知识库信息，对配料表中的每一项进行风险标注。

知识库信息：
{context}

配料表：{question}

请按照以下格式输出（每个配料一行）：
[风险图标] 配料名 — 功能类别
   风险说明：...
   特殊提醒：...

风险图标规则：
🟢 = 安全（天然成分，安全性充分）
🟡 = 注意（合法但存争议或需注意）
🔴 = 回避（已禁用或明确有害）
⚪ = 未知（知识库无数据）
"""


def build_risk_chain(llm, vectorstore):
    """
    构建风险标注 Chain。

    参数:
        llm: LLM 实例
        vectorstore: 向量数据库实例

    返回:
        LangChain Runnable

    使用示例:
        chain = build_risk_chain(llm, vectorstore)
        result = chain.invoke({"question": "安赛蜜、苯甲酸钠、柠檬酸"})
        print(result)
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    prompt_template = _load_prompt_template("risk_label.md")
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        """格式化检索结果"""
        if not docs:
            return "（知识库中暂无相关数据）"

        formatted = []
        for doc in docs:
            meta = doc.metadata
            # 风险标注注重安全信息，精简格式
            formatted.append(
                f"【{meta.get('name', '')}】"
                f"功能:{meta.get('function', '')} | "
                f"风险:{meta.get('risk_level', '')} | "
                f"儿童:{'安全' if meta.get('children_safe') else '注意'} | "
                f"孕妇:{'安全' if meta.get('pregnancy_safe') else '注意'} | "
                f"过敏原:{'、'.join(meta.get('allergens', [])) or '无'} | "
                f"限量:{meta.get('daily_intake_limit', '')} | "
                f"说明:{meta.get('risk_reason', '')}"
            )
        return "\n".join(formatted)

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("风险标注链 (Risk Chain) 构建完成")
    return chain
