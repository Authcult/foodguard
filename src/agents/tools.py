"""
Agent 工具集

将 Chain 包装为 LangChain Tool，供 LangGraph Agent 调用。

每个 Tool 对应一个核心能力：
  - interpret_additives: 解读配料
  - check_risk: 风险标注
  - compare_foods: 对比分析
  - check_allergens: 过敏原检测
"""
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger("foodguard.tools")

# 这些变量在 graph 构建时注入（避免循环依赖）
_interpret_chain = None
_risk_chain = None
_compare_chain = None
_allergy_chain = None
_user_profile = None


def set_chains(interpret=None, risk=None, compare=None, allergy=None):
    """注入 Chain 实例（在 graph 构建时调用）"""
    global _interpret_chain, _risk_chain, _compare_chain, _allergy_chain
    _interpret_chain = interpret
    _risk_chain = risk
    _compare_chain = compare
    _allergy_chain = allergy


def set_user_profile(profile):
    """注入用户画像"""
    global _user_profile
    _user_profile = profile


# ============================================================
# Tool 定义
# ============================================================

@tool
def interpret_additives(question: str, chat_history: str = "") -> str:
    """
    解读食品配料。输入用户关于食品配料的疑问（如配料表文本或具体配料名），
    返回通俗易懂的解读，包括每个配料的作用、安全性、限量等。
    适用场景：用户问"这个是什么"、"XX添加剂安全吗"、"帮我看看这个配料表"。
    """
    if _interpret_chain is None:
        return "错误：解读链未初始化，请先启动系统。"

    logger.info(f"工具调用: interpret_additives, question={question[:80]}...")
    try:
        # 如果有对话历史，附加到问题中提供上下文
        full_question = question
        if chat_history:
            full_question = f"对话历史：\n{chat_history}\n\n当前问题：{question}"
        result = _interpret_chain.invoke({"question": full_question})
        return result
    except Exception as e:
        logger.error(f"interpret_additives 调用失败: {e}")
        return f"解读过程出错：{str(e)}"


@tool
def check_risk(question: str, chat_history: str = "") -> str:
    """
    风险等级标注。输入食品配料表，对每个配料标注风险等级（🟢安全/🟡注意/🔴回避/⚪未知），
    并说明风险原因，特别提醒儿童、孕妇等特殊人群的注意事项。
    适用场景：用户问"这个配料有什么风险"、"哪些要注意"、"这个能给孩子吃吗"。
    """
    if _risk_chain is None:
        return "错误：风险标注链未初始化。"

    logger.info(f"工具调用: check_risk, question={question[:80]}...")
    try:
        full_question = question
        if chat_history:
            full_question = f"对话历史：\n{chat_history}\n\n当前问题：{question}"
        result = _risk_chain.invoke({"question": full_question})
        return result
    except Exception as e:
        logger.error(f"check_risk 调用失败: {e}")
        return f"风险标注过程出错：{str(e)}"


@tool
def compare_foods(food_a: str, food_b: str) -> str:
    """
    对比两款同类食品的配料表。输入两个食品的配料表，从配料数量、
    风险等级分布、添加剂安全性等维度进行对比，推荐更健康的选择。
    适用场景：用户问"哪个更好"、"帮我比较一下这两款"、"哪个更适合孩子"。
    """
    if _compare_chain is None:
        return "错误：对比分析链未初始化。"

    logger.info(f"工具调用: compare_foods")
    try:
        result = _compare_chain.invoke({
            "food_a": food_a,
            "food_b": food_b,
        })
        return result
    except Exception as e:
        logger.error(f"compare_foods 调用失败: {e}")
        return f"对比分析过程出错：{str(e)}"


@tool
def check_allergens(question: str, user_allergens: str = "", chat_history: str = "") -> str:
    """
    过敏原检测。根据用户已知过敏史，检测食品配料表中是否含有危险成分。
    能识别直接匹配、别名匹配和可能的交叉过敏反应。
    适用场景：用户问"这个含大豆吗"、"我对花生过敏能不能吃"。
    """
    if _allergy_chain is None:
        return "错误：过敏检测链未初始化。"

    if not user_allergens and _user_profile:
        user_allergens = ", ".join(_user_profile.known_allergens)

    logger.info(f"工具调用: check_allergens, allergens={user_allergens}")
    try:
        full_question = question
        if chat_history:
            full_question = f"对话历史：\n{chat_history}\n\n当前问题：{question}"
        result = _allergy_chain.invoke({
            "question": full_question,
            "user_allergens": user_allergens or "未提供过敏史",
        })
        return result
    except Exception as e:
        logger.error(f"check_allergens 调用失败: {e}")
        return f"过敏原检测过程出错：{str(e)}"


# 汇总所有工具
ALL_TOOLS = [interpret_additives, check_risk, compare_foods, check_allergens]
