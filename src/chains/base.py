"""
Chain 公共工具模块

提取各 Chain 中重复的 Prompt 加载和文档格式化逻辑。
"""
import logging
from pathlib import Path

from config import PROMPTS_DIR

logger = logging.getLogger("foodguard.chain_base")


def load_prompt_template(filename: str, default_prompt: str = "") -> str:
    """
    从 prompts 目录加载 Markdown 格式的 Prompt 模板。

    参数:
        filename: Prompt 文件名（如 "interpret.md"）
        default_prompt: 文件不存在时的默认 Prompt

    返回:
        Prompt 模板字符串
    """
    prompt_path = PROMPTS_DIR / filename
    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        logger.warning(f"Prompt 文件不存在: {prompt_path}，使用内置默认模板")
        return default_prompt


def split_meta_list(value) -> list[str]:
    """
    将 metadata 中的列表字段安全地拆分为列表。

    ChromaDB 不支持 list 类型的 metadata，loader 中已将列表转为
    逗号分隔字符串（如 "山梨酸,苯甲酸"）。此函数将其还原为列表。

    参数:
        value: 可能是 str、list 或 None

    返回:
        list[str]
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if not value.strip():
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def format_aliases(meta: dict) -> str:
    """格式化别名字段"""
    aliases = split_meta_list(meta.get("aliases", []))
    return "、".join(aliases) if aliases else "无"


def format_allergens(meta: dict) -> str:
    """格式化过敏原字段"""
    allergens = split_meta_list(meta.get("allergens", []))
    return "、".join(allergens) if allergens else "无"
