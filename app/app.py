"""
食鉴（FoodGuard）— Streamlit 主应用

启动方式：
  streamlit run app/app.py

注意：首次运行前需确保已完成以下步骤：
  1. pip install -r requirements.txt
  2. 在 .env 中配置 DEEPSEEK_API_KEY
  3. python scripts/process_data.py
  4. python scripts/build_vectorstore.py
"""
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import streamlit as st
from config import logger


def extract_think(text: str) -> tuple[str, str]:
    """
    从 LLM 输出中分离思考过程和最终回答。

    qwen3 等模型会输出 <think>...</think> 思考过程，
    将其提取出来用于前端折叠展示。

    返回:
        (thinking, answer) 元组
    """
    match = re.search(r"<think>(.*?)</think>\s*", text, flags=re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = text[match.end():].strip()
        return thinking, answer
    return "", text.strip()

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="食鉴 FoodGuard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 初始化系统（缓存，只运行一次）
# ============================================================
@st.cache_resource
def init_system():
    """初始化所有系统组件"""
    logger.info("=" * 60)
    logger.info("初始化食鉴（FoodGuard）系统")
    logger.info("=" * 60)

    from src.models.llm import get_llm
    from src.models.embeddings import get_embeddings
    from src.data.vectorstore import get_vectorstore
    from src.chains.interpret_chain import build_interpret_chain
    from src.chains.risk_chain import build_risk_chain
    from src.chains.compare_chain import build_compare_chain
    from src.chains.allergy_chain import build_allergy_chain
    from src.agents.graph import build_graph
    from src.memory.user_profile import load_profile

    llm = get_llm()
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    interpret_chain = build_interpret_chain(llm, vectorstore)
    risk_chain = build_risk_chain(llm, vectorstore)
    compare_chain = build_compare_chain(llm, vectorstore)
    allergy_chain = build_allergy_chain(llm, vectorstore)

    profile = load_profile()

    graph = build_graph(
        interpret_chain=interpret_chain,
        risk_chain=risk_chain,
        compare_chain=compare_chain,
        allergy_chain=allergy_chain,
        user_profile=profile,
        llm=llm,
    )

    logger.info("系统初始化完成")

    return {
        "llm": llm,
        "embeddings": embeddings,
        "vectorstore": vectorstore,
        "interpret_chain": interpret_chain,
        "risk_chain": risk_chain,
        "compare_chain": compare_chain,
        "allergy_chain": allergy_chain,
        "graph": graph,
    }


# 初始化
with st.spinner("正在初始化系统..."):
    try:
        system = init_system()
    except Exception as e:
        st.error(f"系统初始化失败: {e}")
        st.error("请检查配置 (.env) 和依赖 (pip install -r requirements.txt)")
        st.stop()

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("食鉴 FoodGuard")
    st.caption("食品配料智能分析助手")

    st.divider()

    # 用户过敏原设置（用 session_state 管理，避免 rerun 循环）
    st.subheader("我的过敏史")
    from src.memory.user_profile import load_profile, save_profile

    common_allergens = [
        "花生", "牛奶（乳制品）", "鸡蛋", "大豆",
        "小麦（麸质）", "坚果", "鱼类及海鲜", "甲壳类（虾蟹）",
        "芝麻", "芹菜", "苯丙氨酸", "二氧化硫及亚硫酸盐",
    ]

    # 首次加载：从文件读取到 session_state
    if "allergens" not in st.session_state:
        profile = load_profile()
        st.session_state.allergens = profile.get("known_allergens", [])

    selected = st.multiselect(
        "选择你的过敏原（可选）",
        options=common_allergens,
        default=st.session_state.allergens,
        key="allergen_widget",
        help="选择后，分析配料时会自动检测这些过敏原",
    )

    # 仅在值真正变化时写文件（避免无意义的 rerun）
    if set(selected) != set(st.session_state.allergens):
        st.session_state.allergens = list(selected)
        profile = load_profile()
        profile["known_allergens"] = list(selected)
        save_profile(profile)

    st.divider()

    # 功能说明
    st.subheader("功能")
    st.markdown("""
    - **解读配料** — 解释作用、安全性
    - **风险标注** — 🟢🟡🔴 等级
    - **过敏检测** — 根据过敏史排查
    - **对比分析** — 两款食品比较
    """)

    st.divider()

    st.caption("LLM: Qwen3-4B (Ollama)")
    st.caption("Embedding: BGE-M3 (Ollama)")
    st.caption("DB: ChromaDB")

    st.divider()

    if st.button("清除对话"):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# 主界面
# ============================================================
st.title("食鉴 FoodGuard")
st.caption("输入配料表或提问，系统自动解读配料、标注风险、检测过敏原。")

# 初始化对话历史和会话 ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 助手消息可能包含思考过程
        if msg["role"] == "assistant" and msg.get("thinking"):
            with st.expander("💭 思考过程", expanded=False):
                st.markdown(msg["thinking"])
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("输入配料表或提问（如：安赛蜜和山梨酸钾安全吗？）"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("分析中..."):
            try:
                # 构造 user_profile（从 session_state 取最新过敏原）
                user_profile = {
                    "known_allergens": st.session_state.get("allergens", []),
                    "dietary_preferences": [],
                    "family_members": [],
                }

                result = system["graph"].invoke(
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "user_profile": user_profile,
                    },
                    config={"configurable": {"thread_id": st.session_state.session_id}},
                )

                output = result.get("current_output", "")
                error = result.get("error", "")

                if error:
                    st.error(f"出错了: {error}")
                else:
                    # 分离思考过程和回答
                    thinking, answer = extract_think(output)

                    # 可折叠的思考窗口（类似 DeepSeek 网页版）
                    if thinking:
                        with st.expander("💭 思考过程", expanded=False):
                            st.markdown(thinking)

                    # 最终回答
                    st.markdown(answer)

                    # 存储完整输出到历史（方便导出/调试）
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "thinking": thinking,
                    })
            except Exception as e:
                st.error(f"系统错误: {e}")
                logger.exception("Agent 调用失败")

# ============================================================
# 底部
# ============================================================
st.divider()
st.caption(
    "免责声明：本工具的分析结果基于 AI 模型和 GB2760-2024 数据库，"
    "仅供参考，不构成医疗或食品安全建议。如有特殊健康需求，请咨询专业医师。"
)
