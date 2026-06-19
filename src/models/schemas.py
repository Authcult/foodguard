"""
Pydantic 数据模型

定义项目中所有结构化数据的类型，包括：
  - 添加剂知识库条目
  - 配料分析结果
  - Agent 状态
"""
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
# 添加剂知识库条目（与 JSON 数据对应）
# ============================================================
class AdditiveUsage(BaseModel):
    """添加剂在特定食品中的使用规定"""
    food_category: str = Field(default="", description="食品分类编码，如 01.02.02")
    food_name: str = Field(default="", description="食品类别名称，如 风味发酵乳")
    max_usage: str = Field(default="", description="最大使用量，如 0.35g/kg")
    note: str = Field(default="", description="备注，如 以苯甲酸计")


class AdditiveKnowledge(BaseModel):
    """添加剂知识库条目"""
    name: str = Field(description="中文名称，如 安赛蜜")
    name_en: str = Field(default="", description="英文名称")
    aliases: list[str] = Field(default_factory=list, description="别名列表")
    cns: str = Field(default="", description="CNS编号（中国编码系统）")
    ins: str = Field(default="", description="INS编号（国际编码系统）")
    function: str = Field(default="", description="功能类别，如 甜味剂")
    risk_level: str = Field(
        default="safe",
        description="风险等级：safe(安全) / caution(注意) / avoid(回避)"
    )
    risk_reason: str = Field(default="", description="风险等级的原因说明")
    children_safe: bool = Field(default=True, description="儿童是否安全")
    pregnancy_safe: bool = Field(default=True, description="孕妇是否安全")
    allergens: list[str] = Field(default_factory=list, description="过敏原列表")
    daily_intake_limit: str = Field(default="", description="每日允许摄入量")
    description: str = Field(default="", description="详细描述")
    usages: list[AdditiveUsage] = Field(default_factory=list, description="使用范围列表")


# ============================================================
# 配料分析结果
# ============================================================
class IngredientResult(BaseModel):
    """单个配料的解读结果"""
    name: str = Field(description="配料名称")
    cns: str = Field(default="", description="CNS编号")
    function: str = Field(default="", description="功能，如 防腐剂")
    risk_level: str = Field(default="safe", description="风险等级: safe / caution / avoid / unknown")
    risk_emoji: str = Field(default="🟢", description="风险等级对应的 Emoji")
    risk_reason: str = Field(default="", description="风险说明")
    description: str = Field(default="", description="解读文字")
    allergens: list[str] = Field(default_factory=list, description="含有的过敏原")
    children_safe: bool = Field(default=True, description="儿童是否安全")
    pregnancy_safe: bool = Field(default=True, description="孕妇是否安全")
    daily_intake_limit: str = Field(default="", description="每日摄入限量")


class AllergenAlert(BaseModel):
    """过敏原警报"""
    allergen: str = Field(description="过敏原名称")
    found_in: list[str] = Field(default_factory=list, description="在哪些配料中发现")
    severity: str = Field(default="warning", description="严重程度: warning / danger")


class AnalysisResult(BaseModel):
    """完整的配料分析结果"""
    ingredients: list[IngredientResult] = Field(default_factory=list, description="各配料分析结果")
    allergen_alerts: list[AllergenAlert] = Field(default_factory=list, description="过敏原警报")
    summary: str = Field(default="", description="总体评价")
    suggestions: list[str] = Field(default_factory=list, description="建议")


# ============================================================
# 用户画像
# ============================================================
class UserProfile(BaseModel):
    """用户画像（含过敏史和偏好）"""
    user_id: str = Field(default="default_user", description="用户ID")
    known_allergens: list[str] = Field(default_factory=list, description="已知过敏原")
    dietary_preferences: list[str] = Field(default_factory=list, description="饮食偏好")
    family_members: list[str] = Field(default_factory=list, description="家庭成员类型")
    # 家庭成员类型示例: "儿童", "孕妇", "老人"


# 注意：LangGraph 的 AgentState 使用 TypedDict 定义在 src/agents/graph.py 中，
# 此处不再定义 Pydantic 版本，避免重复和混淆。
