"""
LLM 初始化模块

支持两种后端：
  1. Ollama（本地部署，默认）— 使用 qwen3:4b 等本地模型
  2. DeepSeek（云端 API）— 使用 DeepSeek API

通过 config.LLM_BACKEND 切换。
"""
from config import (
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_TEMPERATURE,
    logger,
)


def get_llm(temperature: float = None):
    """
    获取 LLM 实例。

    根据 config.LLM_BACKEND 自动选择 Ollama 或 DeepSeek。

    参数:
        temperature: 生成温度，默认使用配置值

    返回:
        ChatOllama 或 ChatOpenAI 实例
    """
    if temperature is None:
        temperature = LLM_TEMPERATURE

    if LLM_BACKEND == "ollama":
        return _get_ollama_llm(temperature)
    elif LLM_BACKEND == "deepseek":
        return _get_deepseek_llm(temperature)
    else:
        raise ValueError(f"不支持的 LLM 后端: {LLM_BACKEND}，可选: ollama, deepseek")


def _get_ollama_llm(temperature: float):
    """初始化 Ollama LLM"""
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )
    logger.info(f"LLM 初始化完成 (Ollama): model={OLLAMA_MODEL}, base_url={OLLAMA_BASE_URL}")
    return llm


def _get_deepseek_llm(temperature: float):
    """初始化 DeepSeek LLM（云端 API）"""
    from langchain_openai import ChatOpenAI

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
    )
    logger.info(f"LLM 初始化完成 (DeepSeek): model={DEEPSEEK_MODEL}, base_url={DEEPSEEK_BASE_URL}")
    return llm
