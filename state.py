from typing import TypedDict, Literal, Optional

SkillName = Literal["retrieval", "summary", "comparison", "risk", "synthesis"]

class AgentState(TypedDict, total=False):
    question: str
    skill: SkillName
    route_source: str        # "llm" | "rule"
    route_reason: str
    router_name: Optional[str]
    docs: list               # rag.search 的结果
    context: str
    instruction: str
    answer: str
    sources: list