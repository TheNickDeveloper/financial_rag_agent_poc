# agent_graph.py
"""
LangGraph-based financial research agent.

Graph:
    START -> route -> retrieve -> build_instruction -> generate -> END

Public entry point: FinancialAgentGraph.run(question) -> dict
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict, Literal

from langgraph.graph import StateGraph, START, END

from llm import create_llm
from skills import (
    retrieval_skill,
    summary_skill,
    comparison_skill,
    risk_skill,
    synthesis_skill,
)

from agent import (
    SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    RULE_KEYWORDS,
    SKILL_TIE_BREAK,
    RouteDecision,
    resolve_router_model,
)


SkillName = Literal["retrieval", "summary", "comparison", "risk", "synthesis"]


SKILL_PROMPT_BUILDERS = {
    "retrieval": retrieval_skill,
    "summary": summary_skill,
    "comparison": comparison_skill,
    "risk": risk_skill,
    "synthesis": synthesis_skill,
}


class AgentState(TypedDict, total=False):
    question: str
    skill: SkillName
    route_source: str        # "llm" | "rule"
    route_reason: str
    router_name: Optional[str]
    docs: list
    context: str
    instruction: str
    answer: str
    sources: list


class FinancialAgentGraph:
    """
    LangGraph-based financial research agent.

    Nodes:
        route              -> decide which skill to use
        retrieve           -> pull top_k chunks from the vector store
        build_instruction  -> render the skill-specific prompt
        generate           -> call the LLM and produce the final answer

    Edges are linear. To extend later, replace the edge from 'generate'
    (or 'retrieve') with a conditional edge.
    """

    def __init__(
        self,
        rag,
        provider,
        model,
        top_k: int = 5,
        use_llm_router: bool = True,
        router_provider: Optional[str] = None,
        router_model: Optional[str] = None,
    ):
        self.rag = rag
        self.llm = create_llm(provider, model)
        self.top_k = top_k
        self.use_llm_router = use_llm_router

        self.router = None
        self.router_name = None

        if use_llm_router:
            resolved_router_model = resolve_router_model(
                router_provider or provider,
                router_model,
                model,
            )
            self.router_name = (
                f"{router_provider or provider}:{resolved_router_model}"
            )
            self.router = create_llm(
                router_provider or provider,
                resolved_router_model,
            )

        self.graph = self._build_graph()

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("route", self._node_route)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("build_instruction", self._node_build_instruction)
        builder.add_node("generate", self._node_generate)

        builder.add_edge(START, "route")
        builder.add_edge("route", "retrieve")
        builder.add_edge("retrieve", "build_instruction")
        builder.add_edge("build_instruction", "generate")
        builder.add_edge("generate", END)

        return builder.compile()

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #
    def _node_route(self, state: AgentState) -> dict:
        question = state["question"]

        if self.router is None:
            route = self.rule_route(question)
        else:
            try:
                route = self.llm_route(question)
            except Exception as exc:
                route = self.rule_route(question)
                route["reason"] = (
                    f"LLM router failed ({type(exc).__name__}), "
                    f"fell back to rules"
                )

        return {
            "skill": route["skill"],
            "route_source": route["source"],
            "route_reason": route["reason"],
            "router_name": self.router_name,
        }

    def _node_retrieve(self, state: AgentState) -> dict:
        docs = self.rag.search(state["question"], self.top_k)

        context = "\n\n".join([
            f"[Source: {d['source']} | Page: {d['page']}]\n{d['content']}"
            for d in docs
        ])

        sources = [
            {
                "source": d["source"],
                "page": d["page"],
                "snippet": d["content"][:400],
            }
            for d in docs
        ]

        return {"docs": docs, "context": context, "sources": sources}

    def _node_build_instruction(self, state: AgentState) -> dict:
        builder = SKILL_PROMPT_BUILDERS[state["skill"]]
        instruction = builder(state["question"], state["context"])
        return {"instruction": instruction}

    def _node_generate(self, state: AgentState) -> dict:
        prompt = SYSTEM_PROMPT + "\n\n" + state["instruction"]
        answer = self.llm.invoke(prompt)

        text = answer.content if hasattr(answer, "content") else str(answer)
        return {"answer": text}

    # ------------------------------------------------------------------ #
    # Routing helpers
    # ------------------------------------------------------------------ #
    def rule_route(self, question: str) -> dict:
        q = question.lower()

        scores = {
            skill: sum(1 for k in kws if k in q)
            for skill, kws in RULE_KEYWORDS.items()
        }

        skill = max(
            scores,
            key=lambda s: (scores[s], SKILL_TIE_BREAK[s]),
        )

        return {
            "skill": skill if scores[skill] else "retrieval",
            "source": "rule",
            "reason": (
                f"matched keywords: {scores[skill]}"
                if scores[skill]
                else "no keyword match, defaulting to retrieval"
            ),
        }

    def llm_route(self, question: str) -> dict:
        classifier = self.router.with_structured_output(RouteDecision)

        decision = classifier.invoke(
            ROUTER_SYSTEM_PROMPT
            + "\n\nUser request:\n"
            + question
        )

        return {
            "skill": decision.skill,
            "source": "llm",
            "reason": decision.reason or f"classified as {decision.skill}",
        }

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    def run(self, question: str) -> dict:
        final_state = self.graph.invoke({"question": question})

        return {
            "answer": final_state["answer"],
            "sources": final_state.get("sources", []),
            "skill": final_state["skill"],
            "route": {
                "source": final_state.get("route_source"),
                "reason": final_state.get("route_reason"),
            },
            "router": final_state.get("router_name"),
        }

    # ------------------------------------------------------------------ #
    # Graph visualization
    # ------------------------------------------------------------------ #
    def draw_mermaid(self) -> str:
        """Return a Mermaid diagram string of the compiled graph."""
        return self.graph.get_graph().draw_mermaid()

    def save_graph_image(
        self,
        output_dir: str = "graph_output",
        filename: Optional[str] = None,
        fmt: str = "png",
    ) -> str:
        """
        Render the compiled graph and save it to disk.

        Args:
            output_dir: Folder to save into. Created if missing.
            filename:   Optional file name (without folder). If None,
                        a timestamped name is used.
            fmt:        "png" or "mmd". "png" tries draw_mermaid_png first,
                        and falls back to "mmd" if rendering fails.

        Returns:
            The absolute path of the saved file.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_agent_graph_{stamp}.{fmt}"

        out_path = out_dir / filename

        if fmt == "png":
            try:
                png_bytes = self.graph.get_graph().draw_mermaid_png()
                with open(out_path, "wb") as f:
                    f.write(png_bytes)
                print(f"[graph] PNG saved to {out_path.resolve()}")
                return str(out_path.resolve())
            except Exception as exc:
                # Fall back to .mmd so the user still gets something usable.
                fallback_path = out_path.with_suffix(".mmd")
                mermaid_text = self.graph.get_graph().draw_mermaid()
                with open(fallback_path, "w", encoding="utf-8") as f:
                    f.write(mermaid_text)
                print(
                    f"[graph] PNG rendering failed "
                    f"({type(exc).__name__}: {exc}). "
                    f"Saved Mermaid source instead: "
                    f"{fallback_path.resolve()}"
                )
                return str(fallback_path.resolve())

        # fmt == "mmd"
        mermaid_text = self.graph.get_graph().draw_mermaid()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(mermaid_text)
        print(f"[graph] Mermaid source saved to {out_path.resolve()}")
        return str(out_path.resolve())