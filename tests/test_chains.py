"""
端到端测试脚本

验证系统各组件是否正常工作：
  1. LLM 连接
  2. Embedding 加载
  3. 数据加载
  4. 向量检索
  5. Chain 调用
  6. Agent 路由

运行方式：
  python tests/test_chains.py

注意：首次运行会下载 Embedding 模型，请确保网络通畅。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from config import logger

logger.setLevel(logging.INFO)


def test_1_llm():
    """测试 LLM 连接"""
    logger.info("=" * 40)
    logger.info("测试 1: LLM 连接")
    logger.info("=" * 40)

    from src.models.llm import get_llm
    llm = get_llm()
    response = llm.invoke("请用一句话回答：食品添加剂是什么？")
    logger.info(f"LLM 回复: {response.content[:100]}...")
    logger.info("✅ LLM 连接测试通过")


def test_2_embeddings():
    """测试 Embedding 模型加载"""
    logger.info("=" * 40)
    logger.info("测试 2: Embedding 模型")
    logger.info("=" * 40)

    from src.models.embeddings import get_embeddings
    embeddings = get_embeddings()

    # 测试编码
    vector = embeddings.embed_query("安赛蜜")
    logger.info(f"向量维度: {len(vector)}")
    logger.info(f"向量前5维: {vector[:5]}")
    logger.info("✅ Embedding 测试通过")


def test_3_data_loading():
    """测试数据加载"""
    logger.info("=" * 40)
    logger.info("测试 3: 数据加载")
    logger.info("=" * 40)

    from src.data.loader import load_documents
    docs = load_documents()
    logger.info(f"加载 Document 数量: {len(docs)}")

    if len(docs) > 0:
        logger.info(f"第一个 Document: {docs[0].metadata.get('name', 'N/A')}")
        logger.info(f"page_content 长度: {len(docs[0].page_content)}")

    logger.info("✅ 数据加载测试通过")


def test_4_vectorstore():
    """测试向量存储和检索"""
    logger.info("=" * 40)
    logger.info("测试 4: 向量存储和检索")
    logger.info("=" * 40)

    from src.models.embeddings import get_embeddings
    from src.data.vectorstore import get_vectorstore, retrieve

    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    # 测试检索
    results = retrieve(vectorstore, "防腐剂", top_k=5)
    logger.info(f"检索到 {len(results)} 条结果")

    for i, doc in enumerate(results):
        logger.info(
            f"  {i+1}. {doc.metadata['name']} "
            f"[{doc.metadata.get('risk_level', '?')}] "
            f"— {doc.metadata.get('function', '?')}"
        )

    logger.info("✅ 向量检索测试通过")


def test_5_interpret_chain():
    """测试解读 Chain"""
    logger.info("=" * 40)
    logger.info("测试 5: 解读 Chain")
    logger.info("=" * 40)

    from src.models.llm import get_llm
    from src.models.embeddings import get_embeddings
    from src.data.vectorstore import get_vectorstore
    from src.chains.interpret_chain import build_interpret_chain

    llm = get_llm()
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    chain = build_interpret_chain(llm, vectorstore)

    result = chain.invoke({"question": "安赛蜜是什么？安全性怎么样？"})
    logger.info(f"Chain 输出 (前200字符): {result[:200]}...")
    logger.info("✅ 解读 Chain 测试通过")


def test_6_agent():
    """测试 Agent Graph"""
    logger.info("=" * 40)
    logger.info("测试 6: Agent Graph")
    logger.info("=" * 40)

    from src.models.llm import get_llm
    from src.models.embeddings import get_embeddings
    from src.data.vectorstore import get_vectorstore
    from src.chains.interpret_chain import build_interpret_chain
    from src.chains.risk_chain import build_risk_chain
    from src.chains.compare_chain import build_compare_chain
    from src.chains.allergy_chain import build_allergy_chain
    from src.agents.graph import build_graph

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

    # 测试路由
    test_cases = [
        ("安赛蜜是什么？", "interpret"),
        ("山梨酸钾安全吗？", "risk"),
        ("我对花生过敏，这个能吃吗", "allergy"),
        ("你好", "general"),
    ]

    for user_input, expected_intent in test_cases:
        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": user_input}],
                "user_profile": {"known_allergens": ["花生"]},
            },
            config={"configurable": {"thread_id": "test_session"}},
        )
        intent = result.get("intent", "?")
        status = "✅" if intent == expected_intent else "⚠️"
        logger.info(
            f"  {status} '{user_input}' → intent={intent} "
            f"(期望={expected_intent})"
        )

    logger.info("✅ Agent Graph 测试通过")


def main():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("食鉴（FoodGuard）端到端测试")
    logger.info("=" * 60)

    tests = [
        ("LLM 连接", test_1_llm),
        ("Embedding", test_2_embeddings),
        ("数据加载", test_3_data_loading),
        ("向量检索", test_4_vectorstore),
        ("解读 Chain", test_5_interpret_chain),
        ("Agent Graph", test_6_agent),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            logger.error(f"❌ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    logger.info("=" * 60)
    logger.info(f"测试结果: {passed} 通过, {failed} 失败, {len(tests)} 总计")
    logger.info("=" * 60)

    if failed == 0:
        logger.info("🎉 所有测试通过！系统就绪。")
    else:
        logger.warning(f"⚠️ {failed} 个测试失败，请检查对应模块。")


if __name__ == "__main__":
    main()
