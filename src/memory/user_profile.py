"""
用户画像管理

简单的 JSON 文件存储，记录用户的过敏史和偏好。

在实际使用中，Streamlit 界面会有一个侧边栏让用户填写过敏原信息，
这里用一个 JSON 文件持久化保存。
"""
import json
import logging
from pathlib import Path
from typing import Optional

from config import DATA_DIR

logger = logging.getLogger("foodguard.user_profile")

# 用户画像文件路径
PROFILE_PATH = DATA_DIR / "user_profile.json"


def load_profile() -> dict:
    """
    加载用户画像。

    返回:
        dict，字段：
          - known_allergens: list[str]  已知过敏原
          - dietary_preferences: list[str]  饮食偏好
          - family_members: list[str]  家庭成员类型
    """
    if not PROFILE_PATH.exists():
        return {
            "known_allergens": [],
            "dietary_preferences": [],
            "family_members": [],
        }

    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"加载用户画像失败: {e}，使用默认值")
        return {
            "known_allergens": [],
            "dietary_preferences": [],
            "family_members": [],
        }


def save_profile(profile: dict) -> None:
    """
    保存用户画像到 JSON 文件。

    参数:
        profile: 用户画像字典
    """
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    logger.info(f"用户画像已保存: {profile.get('known_allergens', [])}")


def update_allergens(allergens: list[str]) -> None:
    """更新过敏原列表"""
    profile = load_profile()
    profile["known_allergens"] = allergens
    save_profile(profile)


def add_allergen(allergen: str) -> None:
    """添加一个过敏原"""
    profile = load_profile()
    if allergen and allergen not in profile["known_allergens"]:
        profile["known_allergens"].append(allergen)
        save_profile(profile)
