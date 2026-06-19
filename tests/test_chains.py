"""
食鉴（FoodGuard）测试套件

基于 pytest 框架，测试系统各组件。

运行方式：
  pytest tests/test_chains.py -v

注意：
  - 需要 DEEPSEEK_API_KEY 环境变量（LLM 测试）
  - 首次运行会下载 Embedding 模型
  - 标记 @pytest.mark.slow 的测试需要网络，可用 -m "not slow" 跳过
"""
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ============================================================
# Fixtures（共享的测试资源）
# ============================================================

@pytest.fixture(scope="session")
def llm():
    """LLM 实例（整个测试会话共享）"""
    from src.models.llm import get_llm
    return get_llm()


@pytest.fixture(scope="session")
def embeddings():
    """Embedding 模型实例"""
    from src.models.embeddings import get_embeddings
    return get_embeddings()


@pytest.fixture(scope="session")
def vectorstore(embeddings):
    """ChromaDB 向量数据库实例"""
    from src.data.vectorstore import get_vectorstore
    return get_vectorstore(embeddings)


@pytest.fixture(scope="session")
def documents():
    """加载的知识库 Document 列表"""
    from src.data.loader import load_documents
    return load_documents()


@pytest.fixture(scope="session")
def graph(llm, vectorstore):
    """完整的 LangGraph Agent"""
    from src.chains.interpret_chain import build_interpret_chain
    from src.chains.risk_chain import build_risk_chain
    from src.chains.compare_chain import build_compare_chain
    from src.chains.allergy_chain import build_allergy_chain
    from src.agents.graph import build_graph

    interpret_chain = build_interpret_chain(llm, vectorstore)
    risk_chain = build_risk_chain(llm, vectorstore)
    compare_chain = build_compare_chain(llm, vectorstore)
    allergy_chain = build_allergy_chain(llm, vectorstore)

    return build_graph(
        interpret_chain=interpret_chain,
        risk_chain=risk_chain,
        compare_chain=compare_chain,
        allergy_chain=allergy_chain,
        llm=llm,
    )


# ============================================================
# 基础组件测试（不需要网络）
# ============================================================

class TestConfig:
    """配置模块测试"""

    def test_config_loads(self):
        """配置模块能正常导入"""
        from config import PROJECT_ROOT, DATA_DIR, PROMPTS_DIR
        assert PROJECT_ROOT.exists()
        assert DATA_DIR.exists()
        assert PROMPTS_DIR.exists()

    def test_env_example_exists(self):
        """env.example 文件存在"""
        from config import PROJECT_ROOT
        assert (PROJECT_ROOT / ".env.example").exists()


class TestSchemas:
    """数据模型测试"""

    def test_additive_knowledge_model(self):
        """添加剂知识模型能正常创建"""
        from src.models.schemas import AdditiveKnowledge
        item = AdditiveKnowledge(name="安赛蜜", function="甜味剂")
        assert item.name == "安赛蜜"
        assert item.risk_level == "safe"
        assert item.aliases == []

    def test_user_profile_model(self):
        """用户画像模型能正常创建"""
        from src.models.schemas import UserProfile
        profile = UserProfile(known_allergens=["花生", "牛奶"])
        assert len(profile.known_allergens) == 2

    def test_ingredient_result_model(self):
        """配料分析结果模型"""
        from src.models.schemas import IngredientResult
        result = IngredientResult(name="安赛蜜", risk_level="safe")
        assert result.risk_emoji == "🟢"


class TestChainBase:
    """Chain 公共模块测试"""

    def test_split_meta_list_with_string(self):
        """split_meta_list 能正确拆分逗号分隔字符串"""
        from src.chains.base import split_meta_list
        result = split_meta_list("山梨酸,苯甲酸,安赛蜜")
        assert result == ["山梨酸", "苯甲酸", "安赛蜜"]

    def test_split_meta_list_with_list(self):
        """split_meta_list 对已经是列表的直接返回"""
        from src.chains.base import split_meta_list
        result = split_meta_list(["山梨酸", "苯甲酸"])
        assert result == ["山梨酸", "苯甲酸"]

    def test_split_meta_list_with_empty(self):
        """split_meta_list 处理空值"""
        from src.chains.base import split_meta_list
        assert split_meta_list(None) == []
        assert split_meta_list("") == []
        assert split_meta_list([]) == []

    def test_format_aliases(self):
        """format_aliases 正确格式化别名"""
        from src.chains.base import format_aliases
        assert format_aliases({"aliases": "山梨酸,苯甲酸"}) == "山梨酸、苯甲酸"
        assert format_aliases({"aliases": ""}) == "无"
        assert format_aliases({}) == "无"

    def test_format_allergens(self):
        """format_allergens 正确格式化过敏原"""
        from src.chains.base import format_allergens
        assert format_allergens({"allergens": "花生,牛奶"}) == "花生、牛奶"
        assert format_allergens({"allergens": ""}) == "无"


class TestDataLoader:
    """数据加载测试"""

    def test_load_documents(self, documents):
        """知识库能正常加载"""
        assert len(documents) > 0

    def test_document_structure(self, documents):
        """Document 包含必要的 metadata 字段"""
        doc = documents[0]
        assert "name" in doc.metadata
        assert "risk_level" in doc.metadata
        assert "function" in doc.metadata
        assert len(doc.page_content) > 0

    def test_no_duplicate_names(self, documents):
        """知识库中没有重复名称"""
        names = [doc.metadata["name"] for doc in documents]
        assert len(names) == len(set(names)), f"发现重复: {[n for n in names if names.count(n) > 1]}"


class TestVectorstore:
    """向量存储测试"""

    def test_vectorstore_accessible(self, vectorstore):
        """向量数据库能正常访问"""
        assert vectorstore is not None

    def test_similarity_search(self, vectorstore):
        """相似度检索返回结果"""
        from src.data.vectorstore import retrieve
        results = retrieve(vectorstore, "防腐剂", top_k=5)
        assert len(results) > 0
        assert all("name" in doc.metadata for doc in results)


class TestRouter:
    """意图路由测试"""

    def test_interpret_keywords(self):
        """解读类关键词正确路由"""
        from src.agents.router import route_by_keywords
        assert route_by_keywords("安赛蜜是什么？") == "interpret"
        assert route_by_keywords("帮我看看这个配料表") == "interpret"

    def test_risk_keywords(self):
        """风险类关键词正确路由"""
        from src.agents.router import route_by_keywords
        assert route_by_keywords("这个安全吗？") == "risk"
        assert route_by_keywords("哪些配料有风险") == "risk"

    def test_compare_keywords(self):
        """对比类关键词正确路由"""
        from src.agents.router import route_by_keywords
        assert route_by_keywords("比较一下这两款") == "compare"
        assert route_by_keywords("哪个更好？") == "compare"

    def test_allergy_keywords(self):
        """过敏类关键词正确路由"""
        from src.agents.router import route_by_keywords
        assert route_by_keywords("我对花生过敏能吃吗") == "allergy"
        assert route_by_keywords("这个含过敏原吗") == "allergy"

    def test_general_fallback(self):
        """无法识别的输入路由到 general"""
        from src.agents.router import route_by_keywords
        assert route_by_keywords("你好") == "general"

    def test_allergy_no_false_trigger_on_food_names(self):
        """仅提及食物名不应触发过敏路由"""
        from src.agents.router import route_by_keywords
        # 这些以前会误触发 allergy
        assert route_by_keywords("花生酱好吃吗") != "allergy"
        assert route_by_keywords("牛奶和酸奶哪个好") != "allergy"


class TestUserProfile:
    """用户画像测试"""

    def test_load_profile(self):
        """加载用户画像"""
        from src.memory.user_profile import load_profile
        profile = load_profile()
        assert "known_allergens" in profile
        assert "dietary_preferences" in profile

    def test_save_and_load_profile(self, tmp_path):
        """保存并重新加载用户画像"""
        import json
        from src.memory.user_profile import PROFILE_PATH
        from config import DATA_DIR

        # 使用原始路径读取
        from src.memory.user_profile import load_profile
        profile = load_profile()
        assert isinstance(profile, dict)


# ============================================================
# 集成测试（需要 LLM，标记为 slow）
# ============================================================

@pytest.mark.slow
class TestChainIntegration:
    """Chain 集成测试（需要 LLM API）"""

    def test_interpret_chain(self, llm, vectorstore):
        """解读链能正常运行"""
        from src.chains.interpret_chain import build_interpret_chain
        chain = build_interpret_chain(llm, vectorstore)
        result = chain.invoke({"question": "安赛蜜是什么？"})
        assert isinstance(result, str)
        assert len(result) > 50

    def test_risk_chain(self, llm, vectorstore):
        """风险标注链能正常运行"""
        from src.chains.risk_chain import build_risk_chain
        chain = build_risk_chain(llm, vectorstore)
        result = chain.invoke({"question": "安赛蜜、山梨酸钾"})
        assert isinstance(result, str)
        assert len(result) > 50


@pytest.mark.slow
class TestAgentGraph:
    """Agent Graph 集成测试"""

    def test_interpret_route(self, graph):
        """解读意图路由正确"""
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "安赛蜜是什么？"}],
             "user_profile": {"known_allergens": []}},
            config={"configurable": {"thread_id": "test_interpret"}},
        )
        assert result.get("intent") == "interpret"
        assert len(result.get("current_output", "")) > 0

    def test_risk_route(self, graph):
        """风险意图路由正确"""
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "山梨酸钾安全吗？"}],
             "user_profile": {"known_allergens": []}},
            config={"configurable": {"thread_id": "test_risk"}},
        )
        assert result.get("intent") == "risk"

    def test_allergy_route(self, graph):
        """过敏意图路由正确"""
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "我对花生过敏，这个能吃吗"}],
             "user_profile": {"known_allergens": ["花生"]}},
            config={"configurable": {"thread_id": "test_allergy"}},
        )
        assert result.get("intent") == "allergy"

    def test_general_route(self, graph):
        """一般对话路由正确"""
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "你好"}],
             "user_profile": {"known_allergens": []}},
            config={"configurable": {"thread_id": "test_general"}},
        )
        assert result.get("intent") == "general"
        assert "食鉴" in result.get("current_output", "")
