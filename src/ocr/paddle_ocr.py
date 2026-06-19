"""
OCR 文字识别模块

使用 RapidOCR（PaddleOCR 的 ONNX 移植版）对食品配料表图片进行文字识别。

特点：
  - 不依赖 PaddlePaddle，纯 ONNX Runtime 推理
  - 支持中文识别
  - 轻量级，CPU 即可运行
"""
import logging
from pathlib import Path

logger = logging.getLogger("foodguard.ocr")

# 全局单例
_ocr_instance = None


def get_ocr():
    """
    获取 RapidOCR 实例（单例模式）。

    首次调用会加载 ONNX 模型（约 100MB），后续使用缓存。

    返回:
        RapidOCR 实例
    """
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance

    logger.info("正在初始化 RapidOCR...")
    from rapidocr_onnxruntime import RapidOCR
    _ocr_instance = RapidOCR()
    logger.info("RapidOCR 初始化完成")
    return _ocr_instance


def recognize_text(image_path: str | Path) -> list[dict]:
    """
    识别图片中的文字。

    参数:
        image_path: 图片文件路径

    返回:
        list[dict]，每个 dict 包含：
          - text: 识别出的文本
          - confidence: 置信度 (0-1)
          - box: 文字框坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    ocr = get_ocr()
    image_path = str(image_path)

    logger.info(f"正在识别: {image_path}")
    result, _ = ocr(image_path)

    texts = []
    if result:
        for line in result:
            # RapidOCR 返回格式: [box, text, confidence]
            box = line[0]
            text = line[1]
            confidence = line[2] if len(line) > 2 else 0.0
            texts.append({
                "text": text,
                "confidence": round(float(confidence), 4),
                "box": box,
            })

    logger.info(f"识别完成: 检测到 {len(texts)} 行文本")
    return texts


def extract_ingredient_text(image_path: str | Path) -> str:
    """
    从配料表图片中提取纯文本。

    将 OCR 识别结果按行拼接为连续文本，适合后续送入 LLM 分析。

    参数:
        image_path: 图片文件路径

    返回:
        识别出的配料表文本
    """
    texts = recognize_text(image_path)

    if not texts:
        return ""

    # 按 y 坐标排序（从上到下），同行按 x 坐标排序（从左到右）
    def sort_key(item):
        if item["box"] and len(item["box"]) >= 1:
            box = item["box"]
            # 取左上角 y 坐标
            y = box[0][1] if isinstance(box[0], (list, tuple)) else 0
            x = box[0][0] if isinstance(box[0], (list, tuple)) else 0
            return (y, x)
        return (0, 0)

    sorted_texts = sorted(texts, key=sort_key)

    # 拼接文本，过滤低置信度结果
    lines = []
    for item in sorted_texts:
        if item["confidence"] >= 0.5 and item["text"].strip():
            lines.append(item["text"].strip())

    return "\n".join(lines)
