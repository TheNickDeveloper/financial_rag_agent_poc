import os
from typing import Literal

from pydantic import BaseModel, Field

from llm import create_llm
from skills import (
    retrieval_skill,
    summary_skill,
    comparison_skill,
    risk_skill,
    synthesis_skill
)


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


SKILL_PROMPT_BUILDERS = {
    "retrieval": retrieval_skill,
    "summary": summary_skill,
    "comparison": comparison_skill,
    "risk": risk_skill,
    "synthesis": synthesis_skill
}


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


class FinancialAgent:

    def __init__(self, rag, provider, model, top_k=5,
                 use_llm_router=True, router_provider=None,
                 router_model=None):
        self.rag = rag
        self.llm = create_llm(provider, model)
        self.top_k = top_k
        self.use_llm_router = use_llm_router
        self.router = None
        self.router_name = None

        if use_llm_router:
            router_model = resolve_router_model(
                router_provider or provider,
                router_model,
                model
            )
            self.router_name = f"{router_provider or provider}:{router_model}"
            self.router = create_llm(
                router_provider or provider,
                router_model
            )

    def rule_route(self, question):
        q = question.lower()

        scores = {
            skill: sum(1 for k in kws if k in q)
            for skill, kws in RULE_KEYWORDS.items()
        }

        skill = max(
            scores,
            key=lambda s: (scores[s], SKILL_TIE_BREAK[s])
        )

        return {
            "skill": skill if scores[skill] else "retrieval",
            "source": "rule",
            "reason": (
                f"matched keywords: {scores[skill]}"
                if scores[skill]
                else "no keyword match, defaulting to retrieval"
            )
        }

    def llm_route(self, question):
        classifier = self.router.with_structured_output(RouteDecision)

        decision = classifier.invoke(
            ROUTER_SYSTEM_PROMPT
            + "\n\nUser request:\n"
            + question
        )

        return {
            "skill": decision.skill,
            "source": "llm",
            "reason": decision.reason or f"classified as {decision.skill}"
        }

    def route(self, question):
        if self.router is None:
            return self.rule_route(question)

        try:
            return self.llm_route(question)

        except Exception as exc:
            fallback = self.rule_route(question)
            fallback["reason"] = (
                f"LLM router failed ({type(exc).__name__}), "
                f"fell back to rules"
            )
            return fallback

    def run(self, question):
        route = self.route(question)
        skill = route["skill"]

        docs = self.rag.search(question, self.top_k)

        context = "\n\n".join([
            f"[Source: {d['source']} | Page: {d['page']}]\n{d['content']}"
            for d in docs
        ])

        instruction = SKILL_PROMPT_BUILDERS[skill](question, context)

        prompt = SYSTEM_PROMPT + "\n\n" + instruction

        answer = self.llm.invoke(prompt)

        sources = [
            {
                "source": d["source"],
                "page": d["page"],
                "snippet": d["content"][:400]
            }
            for d in docs
        ]

        return {
            "answer": answer.content if hasattr(answer, "content") else str(answer),
            "sources": sources,
            "skill": skill,
            "route": route,
            "router": self.router_name
        }