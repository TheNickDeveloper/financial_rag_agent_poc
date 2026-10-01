# agent.py
"""
Shared prompts, routing rules and helpers for the financial agent.

The actual agent implementation lives in agent_graph.py (LangGraph).
This module is intentionally free of any agent class so there is only
one way to run the agent.
"""

import os
from typing import Literal

from pydantic import BaseModel, Field


SYSTEM_PROMPT = """
You are a Financial Knowledge Agent.

CRITICAL RULES:
1. Answer ONLY based on the retrieved knowledge provided below.
2. If the retrieved knowledge does NOT contain the answer, respond EXACTLY with:
   "I don't have enough information in the knowledge base to answer this question."
3. Do NOT use your own general knowledge, do NOT infer, do NOT guess.
4. Do NOT cite any source that does not appear in the retrieved knowledge.
"""


ROUTER_SYSTEM_PROMPT = """
You are the request router of a financial research agent.

Classify the user request into exactly one skill:

- retrieval: look up and answer factual questions from the knowledge base
- summary: summarise a report or document, key points (摘要, 重點, summary)
- comparison: compare companies, periods or metrics (比較, 對比, versus, vs)
- risk: analyse risk, downside or uncertainty (風險, 風險分析, risk)
- synthesis: combine insights across several documents into themes and conclusions (綜合, 整體結論, overall)

Reply with the skill name only.
"""


RULE_KEYWORDS = {
    "comparison": [
        "compare", "comparison", "versus", "vs", "differ",
        "比較", "比較", "差異", "對比"
    ],
    "risk": [
        "risk", "risks", "risky", "downside", "threat", "uncertainty",
        "風險", "不確定"
    ],
    "summary": [
        "summary", "summarize", "summarise", "digest", "key points",
        "摘要", "總結", "重點"
    ],
    "synthesis": [
        "overall", "synthes", "theme", "cross-document", "综合", "綜合", "整體"
    ]
}


SKILL_TIE_BREAK = {
    "comparison": 5,
    "risk": 4,
    "summary": 3,
    "synthesis": 2,
    "retrieval": 1
}


DEFAULT_ROUTER_MODELS = {
    "OpenAI": "gpt-4o-mini",
    "DeepSeek": "deepseek-chat",
}


def resolve_router_model(router_provider, router_model, fallback_model):
    if router_model:
        return router_model

    env_model = os.getenv("ROUTER_MODEL")
    if env_model:
        return env_model

    return DEFAULT_ROUTER_MODELS.get(
        router_provider,
        fallback_model
    )


class RouteDecision(BaseModel):
    skill: Literal[
        "retrieval", "summary", "comparison", "risk", "synthesis"
    ] = Field(description="The single skill that best matches the request")
    reason: str = Field(
        default="",
        description="One short sentence explaining the routing decision"
    )