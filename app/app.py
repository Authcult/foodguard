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

import streamlit as st
from config import logger

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

    graph = build_graph(
        interpret_chain=interpret_chain,
        risk_chain=risk_chain,
        compare_chain=compare_chain,
        allergy_chain=allergy_chain,
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

    # 用户过敏原设置
    st.subheader("我的过敏史")
    from src.memory.user_profile import load_profile, update_allergens

    profile = load_profile()
    current_allergens = profile.get("known_allergens", [])

    common_allergens = [
        "花生", "牛奶（乳制品）", "鸡蛋", "大豆",
        "小麦（麸质）", "坚果", "鱼类及海鲜",
        "二氧化硫及亚硫酸盐",
    ]

    selected = st.multiselect(
        "选择你的过敏原（可选）",
        options=common_allergens,
        default=current_allergens,
        help="选择后，分析配料时会自动检测这些过敏原",
    )

    if set(selected) != set(current_allergens):
        update_allergens(selected)

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

    st.caption("LLM: DeepSeek-Chat")
    st.caption("Embedding: BGE-M3")
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

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("输入配料表或提问（如：安赛蜜和山梨酸钾安全吗？）"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("分析中..."):
            try:
                result = system["graph"].invoke(
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "user_profile": load_profile(),
                    },
                    config={"configurable": {"thread_id": "streamlit_session"}},
                )

                output = result.get("current_output", "")
                error = result.get("error", "")

                if error:
                    st.error(f"出错了: {error}")
                else:
                    st.markdown(output)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": output}
                    )
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
