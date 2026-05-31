"""
数据预处理脚本

将 raw 数据 + 人工标注数据 合并处理为知识库格式。

输入：
  1. data/raw/gb2760_additives.json     — 爬虫原始数据
  2. data/raw/manual_annotations.json   — 人工标注（风险等级、安全性等）

输出：
  data/processed/additives_knowledge.json  — 完整知识库
  data/processed/allergens.json            — 过敏原列表

运行方式：
  python scripts/process_data.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from pathlib import Path
from typing import Optional

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, logger


def load_json(filepath: Path) -> list[dict]:
    """加载 JSON 文件，如果不存在则返回空列表"""
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_json(data, filepath: Path) -> None:
    """保存 JSON 文件"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存: {filepath}")


def merge_data(
    crawled: list[dict],
    annotated: list[dict],
) -> list[dict]:
    """
    合并爬虫数据和人工标注数据。

    规则：
      - 以人工标注数据为准（有更丰富的 risk_level、risk_reason 等）
      - 爬虫数据补充 usages 等信息
      - 按名称去重
    """
    # 建立人工标注数据的索引
    annot_index = {}
    for item in annotated:
        name = item.get("name", "")
        if name:
            annot_index[name] = item

    # 合并结果
    result = []
    seen_names = set()

    # 先处理人工标注数据
    for item in annotated:
        name = item.get("name", "")
        if name and name not in seen_names:
            seen_names.add(name)
            # 确保字段完整
            result.append(ensure_all_fields(item))

    # 再处理爬虫数据中人工标注没有的
    for item in crawled:
        name = item.get("name", "")
        if name and name not in seen_names:
            seen_names.add(name)
            # 为爬虫数据补充默认字段
            enriched = enrich_crawled_item(item)
            result.append(enriched)

    logger.info(f"合并完成: {len(result)} 条知识条目")
    logger.info(f"  其中人工标注: {len(annotated)} 条")
    logger.info(f"  爬虫补充: {len(result) - len(annotated)} 条")

    return result


def ensure_all_fields(item: dict) -> dict:
    """确保条目包含所有必需字段，缺失的填默认值"""
    defaults = {
        "name": "",
        "name_en": "",
        "aliases": [],
        "cns": "",
        "ins": "",
        "function": "",
        "risk_level": "safe",       # safe / caution / avoid
        "risk_reason": "",
        "children_safe": True,
        "pregnancy_safe": True,
        "allergens": [],
        "daily_intake_limit": "",
        "description": "",
        "usages": [],
    }
    for key, default in defaults.items():
        if key not in item:
            item[key] = default
    return item


def enrich_crawled_item(item: dict) -> dict:
    """
    为爬虫数据补充默认安全评估。
    爬虫数据只有基本信息，没有风险标注。
    这里给一个保守的默认值，后续可通过人工标注覆盖。
    """
    item = ensure_all_fields(item)
    # 爬虫数据标记为"待审核"
    if not item.get("risk_reason"):
        item["risk_reason"] = "数据来源于 GB2760-2024，尚未进行人工安全性审核"
    return item


def process_allergens(annotated: list[dict]) -> list[dict]:
    """
    从知识库中提取过敏原信息，生成过敏原列表。

    过敏原来自两方面：
      1. 知识库中 allergens 字段非空的添加剂
      2. 常见食物过敏原（见下方 COMMON_ALLERGENS）
    """
    # 从知识库中提取
    allergen_map = {}
    for item in annotated:
        for allergen in item.get("allergens", []):
            if allergen not in allergen_map:
                allergen_map[allergen] = {
                    "name": allergen,
                    "type": "添加剂",
                    "related_additives": [],
                }
            allergen_map[allergen]["related_additives"].append(item["name"])

    # 添加常见食物过敏原
    common_allergens = [
        {
            "name": "花生",
            "type": "食物",
            "aliases": ["花生仁", "花生酱", "花生油"],
            "common_hidden_sources": ["混合坚果", "某些酱料", "烘焙预拌粉"],
        },
        {
            "name": "牛奶（乳制品）",
            "type": "食物",
            "aliases": ["牛奶", "乳清蛋白", "酪蛋白", "乳糖"],
            "common_hidden_sources": ["烘焙食品", "巧克力", "沙拉酱", "加工肉制品"],
        },
        {
            "name": "鸡蛋",
            "type": "食物",
            "aliases": ["蛋黄", "蛋清", "卵蛋白", "溶菌酶"],
            "common_hidden_sources": ["面包", "蛋糕", "面条", "某些疫苗"],
        },
        {
            "name": "大豆",
            "type": "食物",
            "aliases": ["大豆蛋白", "大豆卵磷脂", "酱油", "豆腐"],
            "common_hidden_sources": ["调味酱料", "蛋白粉", "巧克力", "肉制品"],
        },
        {
            "name": "小麦（麸质）",
            "type": "食物",
            "aliases": ["面筋", "谷朊粉", "麦麸", "面粉"],
            "common_hidden_sources": ["酱油", "啤酒", "加工肉制品", "调味料"],
        },
        {
            "name": "坚果",
            "type": "食物",
            "aliases": ["杏仁", "核桃", "腰果", "榛子", "开心果"],
            "common_hidden_sources": ["混合坚果", "巧克力", "烘焙食品", "冰淇淋"],
        },
        {
            "name": "鱼类及海鲜",
            "type": "食物",
            "aliases": ["鱼露", "虾酱", "蚝油", "鱼胶"],
            "common_hidden_sources": ["调味酱料", "汤料", "速冻食品"],
        },
        {
            "name": "二氧化硫及亚硫酸盐",
            "type": "添加剂",
            "aliases": ["二氧化硫", "焦亚硫酸钠", "亚硫酸钠", "亚硫酸氢钠"],
            "common_hidden_sources": ["果脯蜜饯", "葡萄酒", "干制蔬菜", "淀粉"],
        },
    ]

    result = list(allergen_map.values())
    # 合并常见过敏原
    for ca in common_allergens:
        found = False
        for r in result:
            if r["name"] == ca["name"]:
                r.update(ca)
                found = True
                break
        if not found:
            result.append(ca)

    return result


def main():
    """主流程"""
    logger.info("=" * 60)
    logger.info("开始处理数据")
    logger.info("=" * 60)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据源
    crawled = load_json(RAW_DATA_DIR / "gb2760_additives.json")
    annotated = load_json(RAW_DATA_DIR / "manual_annotations.json")

    logger.info(f"爬虫数据: {len(crawled)} 条")
    logger.info(f"人工标注数据: {len(annotated)} 条")

    # 合并
    knowledge = merge_data(crawled, annotated)

    # 保存知识库
    save_json(knowledge, PROCESSED_DATA_DIR / "additives_knowledge.json")

    # 生成过敏原列表
    allergens = process_allergens(annotated)
    save_json(allergens, PROCESSED_DATA_DIR / "allergens.json")

    # 统计
    from collections import Counter
    risk_counts = Counter(item["risk_level"] for item in knowledge)
    logger.info("风险等级分布:")
    logger.info(f"  🟢 安全(safe):    {risk_counts.get('safe', 0)}")
    logger.info(f"  🟡 注意(caution): {risk_counts.get('caution', 0)}")
    logger.info(f"  🔴 回避(avoid):   {risk_counts.get('avoid', 0)}")

    func_counts = Counter(item["function"] for item in knowledge)
    logger.info("功能类别 Top 5:")
    for func, count in func_counts.most_common(5):
        logger.info(f"  {func}: {count}")


if __name__ == "__main__":
    main()
