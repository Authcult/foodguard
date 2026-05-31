"""
爬取 GB2760-2024 食品添加剂使用标准数据

数据来源（按优先级）：
  1. https://2760.foodmate.net  — 食品伙伴网 GB2760 查询系统
  2. https://gb2760.cfsa.net.cn   — 国家食品安全风险评估中心官方数据库

输出：data/raw/gb2760_additives.json

运行方式：
  python scripts/crawl_gb2760.py

注意：
  - 请求间隔 1-2 秒，避免对服务器造成压力
  - 如果某页面爬取失败，会跳过并记录日志
  - 最终会生成一个包含所有成功爬取数据的 JSON 文件
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config import RAW_DATA_DIR, logger

# ============================================================
# 配置
# ============================================================
# 食品伙伴网 GB2760 分类页面
# 各功能类别对应的 category ID
CATEGORY_URLS = {
    "酸度调节剂":       "https://2760.foodmate.net/category/limit/216.html",
    "抗结剂":           "https://2760.foodmate.net/category/limit/217.html",
    "消泡剂":           "https://2760.foodmate.net/category/limit/218.html",
    "抗氧化剂":         "https://2760.foodmate.net/category/limit/219.html",
    "漂白剂":           "https://2760.foodmate.net/category/limit/220.html",
    "膨松剂":           "https://2760.foodmate.net/category/limit/221.html",
    "胶基糖果中基础剂物质": "https://2760.foodmate.net/category/limit/222.html",
    "着色剂":           "https://2760.foodmate.net/category/limit/223.html",
    "护色剂":           "https://2760.foodmate.net/category/limit/224.html",
    "乳化剂":           "https://2760.foodmate.net/category/limit/225.html",
    "酶制剂":           "https://2760.foodmate.net/category/limit/226.html",
    "增味剂":           "https://2760.foodmate.net/category/limit/227.html",
    "面粉处理剂":       "https://2760.foodmate.net/category/limit/228.html",
    "被膜剂":           "https://2760.foodmate.net/category/limit/229.html",
    "水分保持剂":       "https://2760.foodmate.net/category/limit/230.html",
    "防腐剂":           "https://2760.foodmate.net/category/limit/231.html",
    "稳定剂和凝固剂":   "https://2760.foodmate.net/category/limit/232.html",
    "甜味剂":           "https://2760.foodmate.net/category/limit/233.html",
    "增稠剂":           "https://2760.foodmate.net/category/limit/234.html",
    "食品用香料":       "https://2760.foodmate.net/category/limit/235.html",
    "食品工业用加工助剂": "https://2760.foodmate.net/category/limit/236.html",
    "其他":             "https://2760.foodmate.net/category/limit/237.html",
}

REQUEST_DELAY = 1.5  # 请求间隔（秒）
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def fetch_page(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """
    获取页面并返回 BeautifulSoup 对象。
    失败时重试 retries 次。
    """
    for attempt in range(retries):
        try:
            logger.info(f"  请求: {url}")
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.encoding = "utf-8"
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            else:
                logger.warning(f"  HTTP {resp.status_code}，重试 {attempt + 1}/{retries}")
        except requests.RequestException as e:
            logger.warning(f"  请求异常: {e}，重试 {attempt + 1}/{retries}")
            time.sleep(2)
    return None


def parse_list_page(soup: BeautifulSoup, category_function: str) -> list[dict]:
    """
    从分类列表页解析出所有添加剂的名称、CNS、INS、功能类别。

    食品伙伴网 GB2760 列表页实际结构：
      - 页面有多个 <table>
      - 主要数据表：列顺序固定为 添加剂 | 功能 | 最大使用量 | CNS | INS | 备注
      - 有些行是分类标题（如"熟制水产品 09.04"），需要跳过

    返回：list[dict]，每个 dict 包含基本信息
    """
    additives = []

    # 找行数最多的 table（就是数据表，其他 table 只有几行）
    tables = soup.find_all("table")
    if not tables:
        logger.warning("  未找到任何表格")
        return additives

    # 按行数排序，取最大的
    tables_sorted = sorted(tables, key=lambda t: len(t.find_all("tr")), reverse=True)
    table = tables_sorted[0]

    rows = table.find_all("tr")
    logger.info(f"  数据表有 {len(rows)} 行")

    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) < 4:  # 至少需要 添加剂|功能|CNS|INS 四列
            continue

        # 提取文本
        texts = [c.get_text(strip=True).replace("\xa0", " ") for c in cells]

        # 跳过表头行
        first_cell = texts[0]
        if first_cell in ("添加剂", "食品添加剂名称", "中文名称", "名称", ""):
            continue

        # 实际列结构（按列位置取）：
        #   col 0: 添加剂名称
        #   col 1: 功能（可能与 category_function 不同）
        #   col 2: 最大使用量
        #   col 3: CNS 号
        #   col 4: INS 号
        #   col 5: 备注

        name = texts[0]
        actual_function = texts[1] if len(texts) > 1 else category_function
        cns = texts[3] if len(texts) > 3 else ""
        ins = texts[4] if len(texts) > 4 else ""

        # 跳过分类标题行：这些行的特征是没有 CNS 号
        # CNS 号格式如 "19.011" 或 "04.004,04.018"
        if not cns or cns in ("", "—", "-"):
            continue

        # 清理：去掉备注中的干扰
        if "—" in ins or ins == "":
            ins = ins.replace("—", "").strip()

        additives.append({
            "name": name,
            "cns": cns,
            "ins": ins,
            "function": actual_function,
        })

    return additives


def parse_detail_page(soup: BeautifulSoup) -> dict:
    """
    从添加剂详情页解析出使用范围、最大使用量等信息。

    返回：{"usages": [...], "description": ""}
    """
    usages = []
    description = ""

    # 尝试解析使用范围表格
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            texts = [c.get_text(strip=True) for c in cells]
            if len(texts) >= 3 and texts[0] not in ("食品名称", "食品类别", ""):
                # 常见结构：食品类别 | 食品名称 | 最大使用量 | 备注
                usages.append({
                    "food_category": texts[0] if len(texts) > 0 else "",
                    "food_name": texts[1] if len(texts) > 1 else texts[0],
                    "max_usage": texts[2] if len(texts) > 2 else "",
                    "note": texts[3] if len(texts) > 3 else "",
                })

    return {"usages": usages, "description": description}


def crawl_all_categories() -> list[dict]:
    """
    遍历所有功能类别，爬取每类下的添加剂列表。
    """
    all_additives = []

    for function, url in CATEGORY_URLS.items():
        logger.info(f"正在爬取分类: {function}")
        soup = fetch_page(url)
        if not soup:
            logger.warning(f"  跳过分类: {function}（页面获取失败）")
            continue

        items = parse_list_page(soup, function)
        logger.info(f"  找到 {len(items)} 个添加剂")

        # 为每个添加剂尝试获取详情页（如果有链接的话）
        for item in items:
            # 尝试从页面中找详情链接
            # 食品伙伴网的结构通常是 <a href="...">名称</a>
            # 这里我们主要依赖列表页的数据
            pass

        all_additives.extend(items)
        time.sleep(REQUEST_DELAY)

    return all_additives


def crawl_alternative() -> list[dict]:
    """
    备用方案：从 CFSA 官方数据库爬取。
    在食品伙伴网不可用时使用。
    """
    logger.info("尝试备用数据源: gb2760.cfsa.net.cn")

    additives = []
    # CFSA 网站的分类页面 URL 模式
    # 此处的 category ID 可能不同，需要调整
    cfsa_categories = {
        "防腐剂": "http://gb2760.cfsa.net.cn/index.php?a=search&b=3&m=preservatives",
    }

    for function, url in cfsa_categories.items():
        soup = fetch_page(url)
        if soup:
            items = parse_list_page(soup, function)
            additives.extend(items)
            time.sleep(REQUEST_DELAY)

    return additives


def main():
    """主流程：爬取数据并保存到 raw 目录"""
    logger.info("=" * 60)
    logger.info("开始爬取 GB2760-2024 食品添加剂数据")
    logger.info("=" * 60)

    # 确保目录存在
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 主数据源：食品伙伴网
    additives = crawl_all_categories()

    # 如果主数据源获取数据太少，尝试备用数据源
    if len(additives) < 50:
        logger.warning(f"主数据源仅获取到 {len(additives)} 条，尝试备用数据源")
        alt_data = crawl_alternative()
        # 合并去重
        existing_names = {a["name"] for a in additives}
        for a in alt_data:
            if a["name"] not in existing_names:
                additives.append(a)

    # 保存原始数据
    output_path = RAW_DATA_DIR / "gb2760_additives.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(additives, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"爬取完成！共获取 {len(additives)} 条添加剂数据")
    logger.info(f"数据已保存至: {output_path}")
    logger.info("=" * 60)

    # 按功能类别统计
    from collections import Counter
    func_counts = Counter(a["function"] for a in additives)
    logger.info("各类别数量:")
    for func, count in func_counts.most_common():
        logger.info(f"  {func}: {count}")


if __name__ == "__main__":
    main()
