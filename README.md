# 食鉴 (FoodLens) — 基于 RAG + LangGraph + LLM 的食品配料智能分析系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://www.langchain.com/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 **RAG + LangGraph Agent** 的食品配料智能分析工具。输入食品配料表，系统自动解读每种配料的作用、标注风险等级、检测过敏原、对比不同食品。

## 🎯 核心功能

| 功能 | 说明 |
|------|------|
| 🔍 **配料解读** | 逐条解释配料含义、作用、安全性 |
| ⚠️ **风险标注** | 🟢安全 / 🟡注意 / 🔴回避 三级风险标注 |
| ⚡ **过敏原检测** | 根据用户过敏史自动检测危险成分 |
| 📊 **对比分析** | 两款同类食品配料对比，推荐更优选择 |
| 💬 **多轮对话** | 支持追问，Agent 自动识别意图 |

## 🏗️ 系统架构

```
用户输入
    │
    ▼
┌──────────────┐
│  Streamlit   │  前端界面
└──────┬───────┘
       │
┌──────▼───────┐
│  LangGraph   │  Agent 编排
│   Router     │  意图识别 → interpret/risk/compare/allergy
└──────┬───────┘
       │
┌──────▼───────┐
│   Chains     │  RAG 链路
│  检索→Prompt→LLM  │
└──────┬───────┘
       │
┌──────▼───────┐
│  ChromaDB    │  向量数据库
│  BGE-M3      │  Embedding 模型
└──────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/yourusername/foodguard.git
cd foodguard

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY
```

在 [platform.deepseek.com](https://platform.deepseek.com) 获取 API Key。

### 3. 构建知识库

```bash
# 处理数据（手工标注 + 爬虫数据合并）
python scripts/process_data.py

# 构建向量数据库（首次运行会下载 BGE-M3 模型，约 2.2GB）
python scripts/build_vectorstore.py
```

### 4. 启动应用

```bash
streamlit run app/app.py
```

浏览器打开 `http://localhost:8501`。

## 📁 项目结构

```
foodguard/
├── config.py                 # 全局配置
├── .env.example              # 环境变量模板
├── requirements.txt          # Python 依赖
│
├── data/
│   ├── raw/                  # 原始数据（爬虫 + 手工标注）
│   ├── processed/            # 处理后知识库
│   └── chroma_db/            # 向量数据库（持久化）
│
├── scripts/
│   ├── crawl_gb2760.py       # GB2760 数据爬虫
│   ├── process_data.py       # 数据处理与合并
│   └── build_vectorstore.py  # 构建向量库
│
├── src/
│   ├── models/               # LLM & Embedding 初始化
│   │   ├── llm.py            # DeepSeek (ChatOpenAI 兼容)
│   │   ├── embeddings.py     # BGE-M3 (ModelScope/HuggingFace)
│   │   └── schemas.py        # Pydantic 数据模型
│   ├── data/                 # 数据层
│   │   ├── loader.py         # JSON → LangChain Document
│   │   ├── splitter.py       # 文本分割
│   │   └── vectorstore.py    # ChromaDB 封装
│   ├── chains/               # RAG Chain 层
│   │   ├── interpret_chain.py  # 配料解读
│   │   ├── risk_chain.py       # 风险标注
│   │   ├── compare_chain.py    # 对比分析
│   │   └── allergy_chain.py    # 过敏检测
│   ├── agents/               # Agent 编排层
│   │   ├── router.py         # 意图路由（关键词 + LLM fallback）
│   │   ├── tools.py          # Chain → Tool 包装
│   │   └── graph.py          # LangGraph StateGraph
│   ├── memory/               # 用户画像
│   │   └── user_profile.py
│   └── prompts/              # Prompt 模板
│       ├── interpret.md
│       ├── risk_label.md
│       ├── compare.md
│       └── allergy_check.md
│
├── app/
│   └── app.py                # Streamlit 前端
│
└── tests/
    └── test_chains.py         # 端到端测试
```

## 🔧 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| LLM | DeepSeek-Chat | 通过 ChatOpenAI 兼容接口调用 |
| Embedding | BGE-M3 | 1024 维中文向量，ModelScope 下载 |
| 向量数据库 | ChromaDB | 本地持久化，无需额外部署 |
| Agent 框架 | LangGraph | StateGraph 编排，条件边路由 |
| RAG 框架 | LangChain | LCEL 管道语法串联各步骤 |
| 前端 | Streamlit | 纯 Python Web UI |
| 数据采集 | BeautifulSoup4 | GB2760 标准数据爬取 |

## 📊 数据说明

知识库包含 **145 条食品添加剂数据**，来源于：

- **手工标注**（88 条）：含风险等级、安全评估、特殊人群提醒、每日限量
- **GB2760-2024 爬虫**（57 条补充）：CNS/INS 编码、功能分类、使用范围

风险等级分布：

- 🟢 安全：130 条（天然成分或安全性评价充分）
- 🟡 注意：13 条（合法但有争议或需注意）
- 🔴 回避：2 条（已禁用或明确有害）

## 🧪 运行测试

```bash
python tests/test_chains.py
```

测试覆盖 6 个模块：LLM 连接 → Embedding → 数据加载 → 向量检索 → Chain 调用 → Agent 路由。

## ⚠️ 免责声明

本工具的分析结果基于 AI 模型和 GB2760-2024 数据库，仅供参考，不构成医疗或食品安全建议。如有特殊健康需求，请咨询专业医师。

## 📄 License

MIT License
