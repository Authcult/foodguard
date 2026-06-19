"""
配料解读链 (Interpret Chain)

核心 RAG 流程：
  用户输入配料 → 检索知识库 → 拼接 Prompt → LLM 生成解读

这是最简单的 RAG Chain，用于理解 RAG 的基本原理：
  1. Retriever: 从向量库检索相关内容
  2. Prompt: 把检索结果 + 用户问题填入模板
  3. LLM: 生成回答
"""
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.chains.base import load_prompt_template, format_aliases, format_allergens

logger = logging.getLogger("foodguard.interpret_chain")


_DEFAULT_INTERPRET_PROMPT = """你是一位食品配料分析专家。

根据以下知识库信息，用通俗易懂的语言向消费者解读食品配料。

知识库信息：
{context}

用户问题：{question}

请用中文回答，结构清晰，包含每个配料的作用、安全性、限量等信息。
对于知识库中没有的配料，明确说明"暂无数据"。
"""


# ============================================================
# 构建 RAG Chain
# ============================================================
def build_interpret_chain(llm, vectorstore):
    """
    构建配料解读 RAG Chain。

    流程图：
      用户输入 → retriever 检索 → 格式化 Prompt → LLM 生成 → 输出文本

    参数:
        llm: LLM 实例（DeepSeek ChatOpenAI）
        vectorstore: ChromaDB 向量数据库实例

    返回:
        LangChain Runnable（可被 invoke 调用的 Chain）

    使用示例:
        from src.models.llm import get_llm
        from src.data.vectorstore import get_vectorstore
        from src.models.embeddings import get_embeddings
        from src.chains.interpret_chain import build_interpret_chain

        llm = get_llm()
        embeddings = get_embeddings()
        vectorstore = get_vectorstore(embeddings)
        chain = build_interpret_chain(llm, vectorstore)

        result = chain.invoke({"question": "安赛蜜和山梨酸钾是什么？"})
        print(result)
    """
    # 1. 创建 Retriever
    # Retriever 负责：接收查询文本 → 在向量库中搜索 → 返回相关 Document 列表
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 相似度检索
        search_kwargs={"k": 10},   # 返回前 10 条最相似结果
    )

    # 2. 加载 Prompt 模板
    prompt_template = load_prompt_template("interpret.md", _DEFAULT_INTERPRET_PROMPT)
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 3. 格式化检索结果的函数
    # 把检索到的 Document 列表拼接为一个字符串，填入 Prompt 的 {context} 位置
    def format_docs(docs):
        """将检索到的 Document 列表格式化为上下文字符串"""
        if not docs:
            return "（知识库中暂无相关数据）"

        formatted = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            formatted.append(
                f"【条目 {i}】\n"
                f"名称：{meta.get('name', '')}\n"
                f"英文：{meta.get('name_en', '')}\n"
                f"别名：{format_aliases(meta)}\n"
                f"CNS：{meta.get('cns', '')} | INS：{meta.get('ins', '')}\n"
                f"功能：{meta.get('function', '')}\n"
                f"风险等级：{meta.get('risk_level', '')}\n"
                f"风险说明：{meta.get('risk_reason', '')}\n"
                f"儿童安全：{'是' if meta.get('children_safe') else '否'} | "
                f"孕妇安全：{'是' if meta.get('pregnancy_safe') else '否'}\n"
                f"每日限量：{meta.get('daily_intake_limit', '')}\n"
                f"过敏原：{format_allergens(meta)}\n"
                f"---\n{doc.page_content}"
            )
        return "\n\n".join(formatted)

    # 4. 串联 Chain
    # 使用 LCEL（LangChain Expression Language）构建 Pipeline
    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("配料解读链 (Interpret Chain) 构建完成")
    return chain
