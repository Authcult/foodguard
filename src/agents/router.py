"""
意图路由器 (Intent Router)

分析用户输入，判断意图类别，决定调用哪个功能 Chain。

意图类型：
  - interpret:  解读配料（"这个是什么"、"帮我看看配料表"）
  - risk:       风险标注（"安全吗"、"哪些要注意"）
  - compare:    对比分析（"比较一下"、"哪个更好"）
  - allergy:    过敏检测（"含大豆吗"、"我过敏能吃吗"）
  - general:    一般对话（"你好"、"谢谢"）

路由逻辑：
  - 先用简单的关键词匹配做快速路由（省钱、快速）
  - 未来可扩展为 LLM 路由（更智能但更慢、更贵）
"""
import logging
from typing import Literal

logger = logging.getLogger("foodguard.router")

# 意图类型
IntentType = Literal["interpret", "risk", "compare", "allergy", "general"]


def route_by_keywords(user_input: str) -> IntentType:
    """
    基于关键词的快速意图路由。

    规则优先级（从高到低）：
      1. compare: 含"对比""比较""哪个更好""二选一"等关键词
      2. allergy: 含"过敏""过敏原""能吃吗""含...吗"等关键词
      3. risk:    含"安全""风险""有害""有毒""能不能吃""儿童""孕妇"等关键词
      4. interpret: 含"是什么""有什么""配料""帮我看看""解读"等关键词
      5. general:  其他情况

    参数:
        user_input: 用户输入文本

    返回:
        意图类型字符串
    """
    text = user_input.lower().strip()

    # 对比类关键词
    compare_kw = ["对比", "比较", "哪个更好", "哪个好", "二选一", "vs", "pk",
                  "帮我选", "推荐哪个", "选哪个", "哪款", "区别", "差异"]
    if any(kw in text for kw in compare_kw):
        logger.info(f"路由结果: compare (关键词匹配)")
        return "compare"

    # 过敏类关键词（仅意图性关键词，不含食物名，避免误触）
    allergy_kw = ["过敏", "过敏原", "致敏", "能吃吗", "我可以吃",
                  "不能吃", "敢吃", "过敏体质", "不耐受"]
    if any(kw in text for kw in allergy_kw):
        # 排除对比场景（"含...多"可能是对比）
        if "哪个含" not in text and "哪个更" not in text:
            logger.info(f"路由结果: allergy (关键词匹配)")
            return "allergy"

    # 风险类关键词
    risk_kw = ["安全", "风险", "有害", "有毒", "毒性", "致癌", "禁用",
               "能不能吃", "注意", "警惕", "儿童", "孕妇", "孩子", "宝宝",
               "危险", "副作用", "危害", "超标", "不合格"]
    if any(kw in text for kw in risk_kw):
        logger.info(f"路由结果: risk (关键词匹配)")
        return "risk"

    # 解读类关键词
    interpret_kw = ["是什么", "有什么", "配料", "成分", "看看", "解读",
                    "解释", "介绍", "查", "了解", "什么意思", "怎么用",
                    "添加剂", "防腐剂", "色素", "甜味剂"]
    if any(kw in text for kw in interpret_kw):
        logger.info(f"路由结果: interpret (关键词匹配)")
        return "interpret"

    # 默认：一般对话
    logger.info(f"路由结果: general (无明确意图)")
    return "general"


def route_by_llm(user_input: str, llm) -> IntentType:
    """
    基于 LLM 的意图路由（备用，更智能但更慢）。

    当关键词路由不够精确时使用。让 LLM 判断用户意图。

    参数:
        user_input: 用户输入文本
        llm: LLM 实例

    返回:
        意图类型字符串
    """
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template("""\
你是一个意图分类器。分析用户输入，判断意图类别。

类别说明：
- interpret: 用户想了解配料是什么、有什么作用
- risk: 用户关心安全性、风险等级、特殊人群能不能吃
- compare: 用户想对比两款食品
- allergy: 用户关心过敏问题
- general: 打招呼、感谢、闲聊等

用户输入：{input}

请只回复一个单词：interpret / risk / compare / allergy / general""")

    chain = prompt | llm
    response = chain.invoke({"input": user_input})
    intent = response.content.strip().lower()

    valid_intents = {"interpret", "risk", "compare", "allergy", "general"}
    if intent not in valid_intents:
        intent = "general"

    logger.info(f"LLM 路由结果: {intent}")
    return intent
