"""
LLM 初始化模块

使用 DeepSeek API，通过 langchain-openai 的 ChatOpenAI 兼容接口调用。

DeepSeek 兼容 OpenAI 接口格式，只需设置：
  - base_url: https://api.deepseek.com
  - api_key:  你的 DeepSeek API Key
  - model:    deepseek-chat（或 deepseek-reasoner）
"""
import os
from langchain_openai import ChatOpenAI
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_TEMPERATURE,
    logger,
)


def get_llm(temperature: float = None) -> ChatOpenAI:
    """
    获取 DeepSeek LLM 实例。

    参数:
        temperature: 生成温度，默认使用配置值。解读类场景建议 0.1-0.3（更准确），
                     创意类场景建议 0.5-0.7。

    返回:
        ChatOpenAI 实例，实际调用 DeepSeek API。

    使用示例:
        from src.models.llm import get_llm
        llm = get_llm()
        response = llm.invoke("请解释安赛蜜是什么")
    """
    if temperature is None:
        temperature = LLM_TEMPERATURE

    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "DEEPSEEK_API_KEY 未设置！请在 .env 文件中添加：\n"
            "  DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx"
        )

    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
        temperature=temperature,
        # DeepSeek 特有参数
        model_kwargs={
            # deepseek-chat 支持的最大 token 数
            # "max_tokens": 4096,
        },
    )

    logger.info(f"LLM 初始化完成: model={DEEPSEEK_MODEL}, base_url={DEEPSEEK_BASE_URL}")
    return llm
