---
AIGC:
    Label: "1"
    ContentProducer: 001191110102MACQD9K64018705
    ProduceID: 7642551196644884771-data_volume/files/所有对话/主对话/食鉴/CLAUDE.md
    ReservedCode1: ""
    ContentPropagator: 001191110102MACQD9K64028705
    PropagateID: 3612312254548075#1780193709220
    ReservedCode2: ""
---
# 食鉴（FoodGuard）— 项目指引

## 项目概述

开发一个叫"食鉴"的食品配料智能分析Agent。用户输入或拍照食品配料表，系统用自然语言对话式地帮用户解读配料、标注风险、检测过敏原、对比不同食品。

基于 LangChain + LangGraph 的 RAG 应用，Python 开发。

## 核心功能

1. **配料解读**：逐条解释配料含义、作用、安全性
2. **风险标注**：自动标注每个配料的风险等级（🟢安全/🟡注意/🔴回避），并说明原因
3. **过敏原检测**：根据用户保存的过敏史，检测食品中是否含有危险成分
4. **对比分析**：两款同类食品配料对比，推荐更健康选择
5. **儿童/孕妇提醒**：标注不适宜人群
6. **多轮对话**：支持追问、深入讨论，记住上下文

## 技术栈

- Python 3.10+
- LangChain 0.3+（langchain-core, langchain-openai, langchain-community）
- LangGraph（Agent编排）
- ChromaDB（向量数据库，本地持久化）
- BGE-M3 中文 Embedding 模型（通过 HuggingFaceEmbeddings 加载）
- DeepSeek API 作为 LLM（通过 ChatOpenAI 兼容接口调用，base_url 指向 DeepSeek）
- Streamlit（前端界面）
- BeautifulSoup4 + Requests（数据采集）

## 目录结构

```
foodguard/
├── CLAUDE.md                    # 本文件，项目上下文
├── README.md
├── requirements.txt
├── .env                         # DEEPSEEK_API_KEY, OPENAI_API_KEY 等
├── config.py                    # 全局配置
├── data/
│   ├── raw/
│   │   └── gb2760_additives.json
│   ├── processed/
│   │   ├── additives_knowledge.json
│   │   └── allergens.json
│   └── chroma_db/
├── scripts/
│   ├── crawl_gb2760.py
│   ├── process_data.py
│   └── build_vectorstore.py
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm.py              # DeepSeek 初始化（ChatOpenAI 兼容）
│   │   ├── embeddings.py       # BGE-M3 Embedding 初始化
│   │   └── schemas.py          # Pydantic 数据模型
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # 自定义 Document Loader
│   │   ├── splitter.py         # 按配料条目分块
│   │   └── vectorstore.py      # ChromaDB 封装
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── interpret_chain.py  # 配料解读链
│   │   ├── risk_chain.py       # 风险标注链
│   │   ├── compare_chain.py    # 对比分析链
│   │   └── allergy_chain.py    # 过敏检测链
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── router.py           # LangGraph 路由
│   │   ├── tools.py            # Agent 工具集
│   │   └── graph.py            # LangGraph 状态图
│   ├── memory/
│   │   ├── __init__.py
│   │   └── user_profile.py     # 用户画像
│   └── prompts/
│       ├── interpret.md
│       ├── risk_label.md
│       ├── compare.md
│       └── allergy_check.md
├── app/
│   ├── app.py
│   ├── components/
│   │   ├── chat.py
│   │   ├── result_card.py
│   │   └── profile.py
│   └── static/
└── tests/
    ├── test_chains.py
    ├── test_agents.py
    └── test_data.py
```

## 数据说明

### 添加剂知识库 JSON 结构
```json
{
  "name": "安赛蜜",
  "name_en": "acesulfame potassium",
  "aliases": ["乙酰磺胺酸钾"],
  "cns": "19.011",
  "ins": "950",
  "function": "甜味剂",
  "risk_level": "caution",
  "risk_reason": "人工甜味剂，部分研究质疑长期安全性，但符合国标用量内安全",
  "children_safe": true,
  "pregnancy_safe": true,
  "allergens": [],
  "daily_intake_limit": "15mg/kg体重",
  "description": "一种人工甜味剂，甜度约为蔗糖的200倍，广泛用于饮料、糖果、烘焙食品等。在国标规定用量内安全，但部分消费者选择避免。",
  "usages": [
    {
      "food_category": "01.02.02",
      "food_name": "风味发酵乳",
      "max_usage": "0.35g/kg",
      "note": ""
    }
  ]
}
```

risk_level 取值："safe"（天然/无害）、"caution"（合法但争议）、"avoid"（非法或明确有害）

### 过敏原数据
常见食物过敏原列表，含名称、别名、常见隐藏来源。

## 架构设计

### RAG 流程
用户输入 → LangGraph Router 识别意图 → 选择对应 Chain → Chain 内部：构造检索 Query → ChromaDB 检索相关配料 → 拼接 Prompt → LLM 生成回答

### LLM 接入方式
使用 DeepSeek API，通过 langchain-openai 的 ChatOpenAI 兼容接口调用：
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3
)
```

### Agent 设计
使用 LangGraph StateGraph：
- State：messages + intent + user_profile + results
- Router 节点：判断意图（interpret/risk/compare/allergy）
- 功能节点：调用对应 Chain
- 条件边：Router → 不同功能节点

### Memory 设计
- 对话记忆：ConversationBufferWindowMemory，保留最近10轮
- 用户画像：JSON 文件，含过敏原、饮食偏好、家庭成员

## 开发阶段

### Phase 1：基础 RAG 链
config.py → crawl_gb2760.py → process_data.py → schemas.py → llm.py → embeddings.py → loader.py → splitter.py → vectorstore.py → build_vectorstore.py → interpret.md → interpret_chain.py
验收：输入配料表文本，逐条解释每个配料

### Phase 2：对话 + 风险标注
risk_label.md → risk_chain.py → user_profile.py → 对话记忆集成
验收：多轮对话 + 每条配料带风险等级标注

### Phase 3：Agent 模式
compare_chain.py → allergy_chain.py → tools.py → graph.py → router.py
验收：自由提问，Agent 自动选择工具组合

### Phase 4：前端
app.py → chat.py → result_card.py → profile.py
验收：完整可演示的 Web 应用

## 代码风格

- 所有函数参数和返回值必须有类型注解
- 所有公开函数必须有中文 docstring
- 关键路径必须有 try-except，使用 logging 记录错误
- API Key、模型名、路径等全部放在 config.py 或 .env，不硬编码
- 注释用中文写

## 注意事项

1. LLM 使用 DeepSeek，通过 ChatOpenAI 兼容接口，base_url 设为 https://api.deepseek.com
2. ChromaDB 持久化到 data/chroma_db/，不要每次重启都重建
3. Embedding 模型首次加载会下载，后续用缓存
4. Prompt 模板放在独立 .md 文件中，方便迭代
5. .env 中存放 DEEPSEEK_API_KEY，不要提交到 Git
6. 每完成一个 Phase 先测试跑通，再进下一个

---

> 本内容由 Coze AI 生成，请遵循相关法律法规及《人工智能生成合成内容标识办法》使用与传播。
