from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from .plans import LLMSinglePlan


class Reaction(Enum):
    CONTINUE = "cotinue"
    POSTPONE = "postpone"
    CANCEL = "cancel"

class LLMReactionResponse(BaseModel):
    reaction: Reaction = Field(
        description="对该信息的反应。必须是 'cotinue'、'postpone' 或 'cancel' 中的一个。不要提供其他东西。"
    )
    thought_process: str = Field(
        description="对最近发生的事情进行总结，为什么选择这种反应，如果适用，应该做什么来代替当前的计划。以这种格式表述： '我应该继续/推迟/取消我的计划，因为......'"
    )
    new_plan: Optional[LLMSinglePlan] = Field(
        None,
        description="如果反应是'推迟'，应该包括这个字段，以指定新的计划应该是什么。"
    )
