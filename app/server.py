"""
食鉴（FoodGuard）— FastAPI 后端

提供 SSE 流式接口，支持：
  - 实时 token 级流式输出（思考过程 + 回答）
  - 处理步骤状态提示
  - 用户画像管理
  - 会话管理

启动方式：
  python -m uvicorn app.server:app --host 0.0.0.0 --port 8080
"""
import sys
import os
import re
import json
import uuid
import asyncio
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from config import logger

# ============================================================
# 全局系统初始化
# ============================================================
_system = None


def init_system():
    """初始化所有系统组件（只执行一次）"""
    global _system
    if _system is not None:
        return _system

    logger.info("正在初始化食鉴系统...")

    from src.models.llm import get_llm
    from src.models.embeddings import get_embeddings
    from src.data.vectorstore import get_vectorstore
    from src.chains.interpret_chain import build_interpret_chain
    from src.chains.risk_chain import build_risk_chain
    from src.chains.compare_chain import build_compare_chain
    from src.chains.allergy_chain import build_allergy_chain
    from src.agents.router import route_by_keywords, route_by_llm

    llm = get_llm()
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    interpret_chain = build_interpret_chain(llm, vectorstore)
    risk_chain = build_risk_chain(llm, vectorstore)
    compare_chain = build_compare_chain(llm, vectorstore)
    allergy_chain = build_allergy_chain(llm, vectorstore)

    _system = {
        "llm": llm,
        "vectorstore": vectorstore,
        "chains": {
            "interpret": interpret_chain,
            "risk": risk_chain,
            "compare": compare_chain,
            "allergy": allergy_chain,
        },
        "router_keywords": route_by_keywords,
        "router_llm": route_by_llm,
    }

    logger.info("系统初始化完成")
    return _system


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(title="食鉴 FoodGuard", version="0.1.0")

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def startup():
    init_system()


# ============================================================
# 页面路由
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=(static_dir / "index.html").read_text(encoding="utf-8"))


# ============================================================
# API 路由
# ============================================================

@app.post("/api/session")
async def create_session():
    return {"session_id": str(uuid.uuid4())}


@app.get("/api/profile")
async def get_profile():
    from src.memory.user_profile import load_profile
    return load_profile()


@app.put("/api/profile")
async def update_profile(request: Request):
    from src.memory.user_profile import save_profile
    data = await request.json()
    save_profile(data)
    return {"ok": True}


@app.post("/api/ocr")
async def ocr_recognize(file: UploadFile = File(...)):
    """
    图片 OCR 识别端点。

    接收配料表图片，返回识别出的文本。
    """
    from src.ocr.paddle_ocr import extract_ingredient_text

    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = extract_ingredient_text(tmp_path)
        return {"text": text, "filename": file.filename}
    except Exception as e:
        logger.exception("OCR 识别失败")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ============================================================
# SSE 流式对话端点（真正的 token 级流式）
# ============================================================

@app.post("/api/chat")
async def chat(request: Request):
    """
    SSE 流式对话端点。

    流程：路由意图 → 直接调用 chain.stream() → 逐 token 推送

    SSE 事件：
      - event: status    → {"step": "...", "detail": "..."}
      - event: thinking  → {"token": "..."}
      - event: answer    → {"token": "..."}
      - event: done      → {"intent": "..."}
      - event: error     → {"message": "..."}
    """
    data = await request.json()
    message = data.get("message", "").strip()

    if not message:
        return JSONResponse({"error": "消息不能为空"}, status_code=400)

    system = init_system()

    async def event_generator():
        try:
            # ---- 阶段1：意图识别（快，关键词匹配） ----
            yield _sse("status", {"step": "识别意图", "detail": "分析您的问题..."})

            intent = system["router_keywords"](message)

            # 关键词没匹配到，用 LLM 兜底
            if intent == "general":
                try:
                    intent = system["router_llm"](message, system["llm"])
                except Exception:
                    pass

            intent_labels = {
                "interpret": "配料解读",
                "risk": "风险标注",
                "compare": "对比分析",
                "allergy": "过敏检测",
                "general": "一般对话",
            }
            label = intent_labels.get(intent, intent)
            yield _sse("status", {"step": "意图识别完成", "detail": f"识别为：{label}"})

            # ---- 阶段2：查询知识库 + 流式生成 ----
            yield _sse("status", {"step": "查询知识库", "detail": "检索相关配料信息..."})

            chain = system["chains"].get(intent)

            if chain is None:
                # general 意图，直接回复
                reply = _general_reply(message)
                yield _sse("status", {"step": "生成回答", "detail": ""})
                for chunk in _split_text(reply):
                    yield _sse("answer", {"token": chunk})
                    await asyncio.sleep(0.015)
                yield _sse("done", {"intent": intent})
                return

            # ---- 阶段3：chain.stream() 真正的 token 级流式 ----
            yield _sse("status", {"step": "生成回答", "detail": "LLM 正在生成..."})

            chain_input = {"question": message}

            # 对比链需要两个输入
            if intent == "compare":
                food_a, food_b = _split_foods(message)
                chain_input = {"food_a": food_a, "food_b": food_b}

            # 过敏链需要过敏原信息
            if intent == "allergy":
                from src.memory.user_profile import load_profile
                allergens = load_profile().get("known_allergens", [])
                chain_input["user_allergens"] = ", ".join(allergens) if allergens else ""

            # 关键：用 chain.stream() 逐 token 输出
            in_think = False
            think_buf = ""
            answer_buf = ""

            for token in chain.stream(chain_input):
                # 检测 thinking 标签
                if "<think>" in token and not in_think:
                    in_think = True
                    # <think> 之前的内容算 answer
                    before = token.split("<think>")[0]
                    if before:
                        answer_buf += before
                        yield _sse("answer", {"token": before})
                    think_buf = ""
                    continue

                if in_think:
                    if "</think>" in token:
                        in_think = False
                        after = token.split("</think>", 1)[1]
                        think_buf += token.split("</think>")[0]
                        # 输出完整的思考过程
                        yield _sse("thinking", {"token": think_buf})
                        if after:
                            answer_buf += after
                            yield _sse("answer", {"token": after})
                        think_buf = ""
                    else:
                        think_buf += token
                else:
                    answer_buf += token
                    yield _sse("answer", {"token": token})

                await asyncio.sleep(0)  # 让出事件循环

            yield _sse("done", {"intent": intent})

        except Exception as e:
            logger.exception("SSE 流处理异常")
            yield _sse("error", {"message": str(e)})

    return EventSourceResponse(event_generator())


def _sse(event: str, data: dict) -> dict:
    """构造 SSE 事件"""
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


def _general_reply(message: str) -> str:
    """一般对话的简单回复"""
    text = message.lower().strip()
    greetings = ["你好", "hi", "hello", "嗨", "在吗", "hey"]
    thanks = ["谢谢", "感谢", "thanks", "thank", "辛苦了"]

    if any(g in text for g in greetings):
        return (
            "你好！👋 我是**食鉴（FoodGuard）**，你的食品配料智能分析助手。\n\n"
            "我可以帮你：\n"
            "🔍 **解读配料** — 拍个配料表发给我，或直接输入配料名称\n"
            "⚠️  **风险标注** — 标注每个配料的🟢🟡🔴风险等级\n"
            "⚡ **过敏检测** — 告诉我你的过敏史，我帮你排查\n"
            "📊 **对比分析** — 两款食品比比看，推荐更健康的\n\n"
            "直接告诉我你想了解什么吧！"
        )
    if any(t in text for t in thanks):
        return "不客气！随时找我聊食品配料 😊"
    return (
        "我是食品配料分析助手，可以帮你解读配料、标注风险、检测过敏原、对比食品。"
        "请告诉我你想了解什么？"
    )


def _split_foods(message: str) -> tuple:
    """从对比请求中提取两个食品"""
    for sep in ["\n\n", " vs ", " VS ", " 和 ", " 跟 ", " 与 "]:
        if sep in message:
            parts = message.split(sep, 1)
            if len(parts) == 2 and len(parts[0].strip()) > 5 and len(parts[1].strip()) > 5:
                return parts[0].strip(), parts[1].strip()
    return message, message


def _split_text(text: str):
    """按自然断点分块"""
    paragraphs = text.split("\n")
    for i, para in enumerate(paragraphs):
        if not para.strip():
            yield "\n"
            continue
        segments = re.split(r'([。！？；：，、])', para)
        for seg in segments:
            if seg:
                yield seg
        if i < len(paragraphs) - 1:
            yield "\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
