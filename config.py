"""
食鉴（FoodGuard）全局配置
所有 API Key、模型名、路径等集中管理，不硬编码到业务代码中。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts"

# ============================================================
# LLM 配置 (DeepSeek)
# ============================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# ============================================================
# Embedding 模型配置 (BGE-M3)
# ============================================================
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # 本地开发用 cpu

# ============================================================
# ChromaDB 配置
# ============================================================
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "food_additives")

# ============================================================
# 检索配置
# ============================================================
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))  # 检索返回条数

# ============================================================
# 对话记忆配置
# ============================================================
CONVERSATION_MAX_ROUNDS = int(os.getenv("CONVERSATION_MAX_ROUNDS", "10"))

# ============================================================
# 日志
# ============================================================
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("foodguard")
