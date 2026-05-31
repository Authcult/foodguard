"""
LangGraph Agent 状态图

这是整个 Agent 的"大脑"，用 LangGraph 的 StateGraph 编排流程：

  用户输入 → Router（意图识别）→ 功能节点（调用 Chain）→ 输出回答

LangGraph 核心概念：
  - State: 在节点间传递的数据，包含消息历史、意图、结果等
  - Node: 执行具体逻辑的函数（路由、解读、风险标注...）
  - Edge: 节点之间的连接方向
  - Conditional Edge: 根据条件选择下一个节点（Router 的核心）

流程图：
                    ┌─────────────────┐
                    │  用户输入        │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Router 节点    │
                    │  (识别意图)      │
                    └───┬───┬───┬───┬─┘
                        │   │   │   │
              ┌─────────┘   │   │   └──────────┐
              │             │   │              │
    interpret  risk     compare  allergy    general
              │             │   │              │
              └─────────┬───┘   └──────────────┘
                        │
              ┌─────────▼──────────┐
              │   最终回答          │
              └────────────────────┘
"""
import logging
from typing import TypedDict, Optional, Annotated
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.models.schemas import UserProfile
from src.agents.router import route_by_keywords, route_by_llm
from src.agents.tools import (
    set_chains,
    set_user_profile,
    interpret_additives,
    check_risk,
    compare_foods,
    check_allergens,
)

logger = logging.getLogger("foodguard.graph")

# LLM 实例（build_graph 时注入，用于路由 fallback）
_llm = None


# ============================================================
# State 定义
# ============================================================
class AgentState(TypedDict):
    """
    LangGraph Agent 的状态，在所有节点间共享。

    使用 TypedDict（而非 Pydantic）是因为 LangGraph 对 TypedDict 的
    增量更新支持更好（每个节点只需要返回更新的字段）。
    """
    messages: Annotated[list, operator.add]  # 对话历史，自动追加
    intent: str                              # 当前意图
    user_profile: dict                       # 用户画像（过敏史等）
    current_output: str                      # 当前节点的输出
    error: str                               # 错误信息


# ============================================================
# 节点函数
# ============================================================
# 每个节点函数：
#   接收当前 State → 执行逻辑 → 返回 State 的部分更新

def router_node(state: AgentState) -> dict:
    """
    路由节点：分析最新消息，识别用户意图。

    从 messages 中取最新一条作为当前输入，调用 router 判断意图。
    """
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "general", "error": ""}

    # 最新消息
    last_msg = messages[-1]
    # 提取文本内容（可能是字符串或 LangChain Message 对象）
    if hasattr(last_msg, "content"):
        user_input = last_msg.content
    else:
        user_input = str(last_msg)

    intent = route_by_keywords(user_input)

    # 关键词没匹配到时，用 LLM 兜底
    if intent == "general" and _llm is not None:
        try:
            intent = route_by_llm(user_input, _llm)
        except Exception as e:
            logger.warning(f"LLM 路由失败，使用默认值: {e}")

    logger.info(f"路由: '{user_input[:50]}...' → {intent}")

    return {"intent": intent, "error": ""}


def interpret_node(state: AgentState) -> dict:
    """解读节点"""
    messages = state.get("messages", [])
    user_input = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    try:
        output = interpret_additives.invoke({"question": user_input})
        return {"current_output": str(output), "error": ""}
    except Exception as e:
        return {"current_output": "", "error": str(e)}


def risk_node(state: AgentState) -> dict:
    """风险标注节点"""
    messages = state.get("messages", [])
    user_input = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    try:
        output = check_risk.invoke({"question": user_input})
        return {"current_output": str(output), "error": ""}
    except Exception as e:
        return {"current_output": "", "error": str(e)}


def compare_node(state: AgentState) -> dict:
    """
    对比分析节点。

    对比场景特殊：用户输入可能包含两个食品的配料表。
    简单处理：将整个输入作为 food_a 和 food_b 传给 compare chain，
    让 LLM 自己去解析。未来可以改进为更结构化的输入。
    """
    messages = state.get("messages", [])
    user_input = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    try:
        # 简单策略：尝试按"食品A"/"食品B"或"第一款"/"第二款"或空行分割
        # 如果无法分割，就把整个输入同时作为 food_a 和 food_b（LLM 会自己处理）
        output = compare_foods.invoke({
            "food_a": user_input,
            "food_b": user_input,
        })
        return {"current_output": str(output), "error": ""}
    except Exception as e:
        return {"current_output": "", "error": str(e)}


def allergy_node(state: AgentState) -> dict:
    """过敏检测节点"""
    messages = state.get("messages", [])
    user_input = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    profile = state.get("user_profile", {})
    user_allergens = profile.get("known_allergens", [])

    try:
        output = check_allergens.invoke({
            "question": user_input,
            "user_allergens": ", ".join(user_allergens) if user_allergens else "",
        })
        return {"current_output": str(output), "error": ""}
    except Exception as e:
        return {"current_output": "", "error": str(e)}


def general_node(state: AgentState) -> dict:
    """一般对话节点（招呼、感谢等，不需要调用 Chain）"""
    messages = state.get("messages", [])
    user_input = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    # 简单回复，不做 RAG
    greetings = ["你好", "hi", "hello", "嗨", "在吗", "hey"]
    thanks = ["谢谢", "感谢", "thanks", "thank", "辛苦了"]

    text = user_input.lower().strip()
    if any(g in text for g in greetings):
        reply = (
            "你好！👋 我是**食鉴（FoodGuard）**，你的食品配料智能分析助手。\n\n"
            "我可以帮你：\n"
            "🔍 **解读配料** — 拍个配料表发给我，或直接输入配料名称\n"
            "⚠️  **风险标注** — 标注每个配料的🟢🟡🔴风险等级\n"
            "⚡ **过敏检测** — 告诉我你的过敏史，我帮你排查\n"
            "📊 **对比分析** — 两款食品比比看，推荐更健康的\n\n"
            "直接告诉我你想了解什么吧！"
        )
    elif any(t in text for t in thanks):
        reply = "不客气！随时找我聊食品配料 😊"
    else:
        reply = (
            "我是食品配料分析助手，可以帮你解读配料、标注风险、检测过敏原、对比食品。"
            "请告诉我你想了解什么？"
        )

    return {"current_output": reply, "error": ""}


# ============================================================
# 条件边：路由到不同节点
# ============================================================
def route_by_intent(state: AgentState) -> str:
    """
    根据 Router 输出的 intent，决定下一个节点。

    这个函数是"条件边"的核心：
      LangGraph 在 Router 节点执行完后调用这个函数，
      根据返回的字符串决定走向哪个节点。
    """
    intent = state.get("intent", "general")
    route_map = {
        "interpret": "interpret_node",
        "risk": "risk_node",
        "compare": "compare_node",
        "allergy": "allergy_node",
        "general": "general_node",
    }
    return route_map.get(intent, "general_node")


# ============================================================
# 构建 Graph
# ============================================================
def build_graph(
    interpret_chain=None,
    risk_chain=None,
    compare_chain=None,
    allergy_chain=None,
    user_profile: UserProfile = None,
    llm=None,
):
    """
    构建 LangGraph StateGraph。

    参数:
        interpret_chain: 配料解读 Chain
        risk_chain: 风险标注 Chain
        compare_chain: 对比分析 Chain
        allergy_chain: 过敏检测 Chain
        user_profile: 用户画像

    返回:
        编译后的 LangGraph Graph（Runnable）

    使用示例:
        from src.agents.graph import build_graph
        graph = build_graph(interpret_chain, risk_chain, compare_chain, allergy_chain)

        # 运行
        result = graph.invoke({
            "messages": [{"role": "user", "content": "安赛蜜安全吗？"}],
            "user_profile": {"known_allergens": []},
        })
        print(result["current_output"])
    """
    # 0. 存储 LLM 实例（供 router_node fallback 使用）
    global _llm
    _llm = llm

    # 1. 注入依赖到工具层
    set_chains(
        interpret=interpret_chain,
        risk=risk_chain,
        compare=compare_chain,
        allergy=allergy_chain,
    )
    if user_profile:
        set_user_profile(user_profile)

    # 1. 创建 StateGraph
    workflow = StateGraph(AgentState)

    # 2. 添加节点
    workflow.add_node("router_node", router_node)
    workflow.add_node("interpret_node", interpret_node)
    workflow.add_node("risk_node", risk_node)
    workflow.add_node("compare_node", compare_node)
    workflow.add_node("allergy_node", allergy_node)
    workflow.add_node("general_node", general_node)

    # 3. 设置入口
    workflow.set_entry_point("router_node")

    # 4. 添加条件边：Router → 功能节点
    workflow.add_conditional_edges(
        "router_node",
        route_by_intent,
        {
            "interpret_node": "interpret_node",
            "risk_node": "risk_node",
            "compare_node": "compare_node",
            "allergy_node": "allergy_node",
            "general_node": "general_node",
        },
    )

    # 5. 功能节点 → 结束
    workflow.add_edge("interpret_node", END)
    workflow.add_edge("risk_node", END)
    workflow.add_edge("compare_node", END)
    workflow.add_edge("allergy_node", END)
    workflow.add_edge("general_node", END)

    # 6. 编译（带内存检查点，用于记录对话历史）
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    logger.info("LangGraph Agent 构建完成")
    logger.info(f"节点: {list(workflow.nodes.keys())}")

    return graph
