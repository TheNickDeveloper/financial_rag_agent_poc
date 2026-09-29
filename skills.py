def retrieval_skill(question, context):
    return f"""
Skill: Knowledge Retrieval

User question:
{question}

Retrieved knowledge:
{context}

Answer the question using the retrieved knowledge.
Use concise sections and cite source names inline.
"""


def summary_skill(question, context):
    return f"""
Skill: Financial Document Summary

User request:
{question}

Retrieved knowledge:
{context}

Produce:
1. Executive Summary
2. Key Financial Facts
3. Business / Strategic Points
4. Risks or Uncertainties
5. Sources

Do not add facts not present in the retrieved knowledge.
"""


def comparison_skill(question, context):
    return f"""
Skill: Company / Financial Comparison

User request:
{question}

Retrieved knowledge:
{context}

Create a neutral comparison table where the retrieved information supports it.

Compare:
- Business model
- Revenue / growth
- Profitability
- Key products
- Competitive position
- Risks
- Other relevant metrics

Do not rank companies or declare a winner.
If a metric is unavailable, state "Not available in retrieved documents".
"""


def synthesis_skill(question, context):
    return f"""
Skill: Cross-Document Research Synthesis

User request:
{question}

Retrieved knowledge:
{context}

Produce:
1. Common themes across the retrieved documents
2. Points of agreement and points of conflict between sources
3. Underlying drivers or dynamics behind the numbers
4. What the retrieved documents do not answer
5. Sources

Reconcile contradictions between sources explicitly rather than silently picking one.
Do not introduce facts that are absent from the retrieved knowledge.
"""


def risk_skill(question, context):
    return f"""
Skill: Financial Risk Analysis

User request:
{question}

Retrieved knowledge:
{context}

Structure the answer around:
- Market risk
- Business risk
- Financial risk
- Operational risk
- Regulatory / geopolitical risk
- Key uncertainty

Clearly distinguish documented risks from analytical interpretation.
Do not make investment recommendations.
"""