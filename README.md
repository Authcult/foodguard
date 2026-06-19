# 食鉴 (FoodGuard) — 基于 RAG + LangGraph + LLM 的食品配料智能分析系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://www.langchain.com/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 **RAG + LangGraph Agent** 的食品配料智能分析工具。输入食品配料表（文字或拍照），系统自动解读每种配料的作用、标注风险等级、检测过敏原、对比不同食品。

##  核心功能

| 功能 | 说明 |
|------|------|
|  **配料解读** | 逐条解释配料含义、作用、安全性 |
|  **风险标注** | 🟢安全 / 🟡注意 / 🔴回避 三级风险标注 |
|  **过敏原检测** | 根据 12 种常见过敏史自动检测危险成分 |
|  **对比分析** | 两款同类食品配料对比，推荐更优选择 |
|  **📷 OCR 识别** | 拍照上传配料表图片，自动识别文字并分析 |
|  **多轮对话** | 支持追问，Agent 自动识别意图，带对话记忆 |
|  **💭 思考过程** | 展示模型推理过程（可折叠窗口） |
|  **流式输出** | SSE 实时 token 级流式输出，无需等待 |

##  系统架构

```
用户输入（文字 / 图片）
    │
    ▼
┌──────────────┐
│  FastAPI +    │  前端界面（HTML/CSS/JS）
│  SSE 流式输出  │  实时 token 级推送
└──────┬───────┘
       │
┌──────▼───────┐
│  LangGraph   │  Agent 编排
│   Router     │  意图识别 → interpret/risk/compare/allergy
└──────┬───────┘
       │
┌──────▼───────┐
│   Chains     │  RAG 链路
│  检索→Prompt→LLM  │  chain.stream() 逐 token 流式
└──────┬───────┘
       │
┌──────▼───────┐
│  ChromaDB    │  向量数据库
│  BGE-M3      │  Embedding 模型（Ollama）
└──────────────┘
```

##  快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/CoolGuog/foodguard.git
cd foodguard

# 安装依赖（可用清华源加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置 LLM

默认使用 **Ollama** 本地部署，无需 API Key：

```bash
# 安装 Ollama（https://ollama.com）
# 下载模型
ollama pull qwen3:4b
ollama pull bge-m3:latest
```

如需使用 DeepSeek 云端 API，编辑 `.env`：

```bash
cp .env.example .env
# 设置 LLM_BACKEND=deepseek，填入 DEEPSEEK_API_KEY
```

### 3. 构建知识库

```bash
# 处理数据（手工标注 + 爬虫数据合并）
python scripts/process_data.py

# 构建向量数据库（首次运行会下载 Embedding 模型）
python scripts/build_vectorstore.py
```

### 4. 启动应用

```bash
# FastAPI 版本（推荐，支持流式输出 + OCR）
python -m uvicorn app.server:app --host 0.0.0.0 --port 8080

# 或 Streamlit 版本（备用）
streamlit run app/app.py
```

浏览器打开 `http://localhost:8080`。

##  项目结构

```
foodguard/
├── config.py                 # 全局配置
├── .env.example              # 环境变量模板
├── requirements.txt          # Python 依赖
├── pyproject.toml            # 项目管理（可 pip install -e .）
│
├── data/
│   ├── raw/                  # 原始数据（爬虫 + 手工标注）
│   ├── processed/            # 处理后知识库
│   │   ├── additives_knowledge.json  # 添加剂知识库（145 条）
│   │   └── allergens.json            # 过敏原数据（12 种）
│   └── chroma_db/            # 向量数据库（持久化）
│
├── scripts/
│   ├── crawl_gb2760.py       # GB2760 数据爬虫
│   ├── process_data.py       # 数据处理与合并
│   └── build_vectorstore.py  # 构建向量库
│
├── src/
│   ├── models/               # LLM & Embedding 初始化
│   │   ├── llm.py            # Ollama / DeepSeek 双后端
│   │   ├── embeddings.py     # Ollama / HuggingFace 双后端
│   │   └── schemas.py        # Pydantic 数据模型
│   ├── data/                 # 数据层
│   │   ├── loader.py         # JSON → LangChain Document
│   │   ├── splitter.py       # 文本分割
│   │   └── vectorstore.py    # ChromaDB 封装
│   ├── chains/               # RAG Chain 层
│   │   ├── base.py           # 公共工具（Prompt 加载、文档格式化）
│   │   ├── interpret_chain.py  # 配料解读
│   │   ├── risk_chain.py       # 风险标注
│   │   ├── compare_chain.py    # 对比分析
│   │   └── allergy_chain.py    # 过敏检测
│   ├── agents/               # Agent 编排层
│   │   ├── router.py         # 意图路由（关键词 + LLM fallback）
│   │   ├── tools.py          # Chain → Tool 包装
│   │   └── graph.py          # LangGraph StateGraph
│   ├── ocr/                  # OCR 识别
│   │   └── paddle_ocr.py     # RapidOCR 封装
│   ├── memory/               # 用户画像
│   │   └── user_profile.py
│   └── prompts/              # Prompt 模板
│       ├── interpret.md
│       ├── risk_label.md
│       ├── compare.md
│       └── allergy_check.md
│
├── app/
│   ├── server.py             # FastAPI 后端（SSE 流式）
│   ├── app.py                # Streamlit 前端（备用）
│   └── static/
│       ├── index.html        # 主页面
│       ├── style.css         # 样式（深色主题）
│       └── app.js            # 前端逻辑
│
└── tests/
    └── test_chains.py         # pytest 测试套件
```

##  技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| LLM | Qwen3-4B (Ollama) | 本地部署，支持 DeepSeek 云端切换 |
| Embedding | BGE-M3 (Ollama) | 1024 维中文向量，本地推理 |
| 向量数据库 | ChromaDB | 本地持久化，无需额外部署 |
| Agent 框架 | LangGraph | StateGraph 编排，条件边路由 |
| RAG 框架 | LangChain | LCEL 管道语法，支持 stream() |
| OCR | RapidOCR | PaddleOCR 的 ONNX 移植版，CPU 推理 |
| 后端 | FastAPI + SSE | 流式输出，token 级实时推送 |
| 前端 | HTML/CSS/JS | 深色主题，marked.js 渲染 Markdown |
| 数据采集 | BeautifulSoup4 | GB2760 标准数据爬取 |

##  数据说明

知识库包含 **145 条食品添加剂数据**，来源于：

- **手工标注**（88 条）：含风险等级、安全评估、特殊人群提醒、每日限量
- **GB2760-2024 爬虫**（57 条补充）：CNS/INS 编码、功能分类、使用范围

风险等级分布：

- 🟢 安全：130 条（天然成分或安全性评价充分）
- 🟡 注意：13 条（合法但有争议或需注意）
- 🔴 回避：2 条（已禁用或明确有害）

过敏原覆盖 **12 种**：花生、牛奶、鸡蛋、大豆、小麦、坚果、鱼类、甲壳类、芝麻、芹菜、苯丙氨酸、亚硫酸盐。

##  运行测试

```bash
# 运行所有非网络测试
pytest tests/test_chains.py -v -m "not slow"

# 运行全部测试（需要 LLM 服务）
pytest tests/test_chains.py -v
```

测试覆盖：配置、Schema、ChainBase、数据加载、向量检索、意图路由、用户画像、Chain 集成、Agent Graph。

##  免责声明

本工具的分析结果基于 AI 模型和 GB2760-2024 数据库，仅供参考，不构成医疗或食品安全建议。如有特殊健康需求，请咨询专业医师。

##  License

MIT License
