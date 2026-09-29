# Financial Knowledge Agent — RAG + Agentic Skills POC

> A knowledge agent that reads financial research reports, answers domain-specific questions, and honestly says "I don't know" when the knowledge base doesn't cover it.
> This document breaks down how it works from a workflow perspective, and what value it delivers to the business.

---

## 1. What Problem This POC Solves

Financial research teams face a daily flood of long-form reports — broker research, asset allocation white papers, quantitative strategy articles. This knowledge sits in PDFs and folders. Retrieval relies on keywords. Understanding relies on human memory.

The traditional "keyword search + generic LLM" approach has two fatal flaws:

- **Can't find it**: Keyword matching doesn't understand semantics. Ask "how to improve capital efficiency" and it won't find articles discussing "return stacking."
- **Makes things up**: A generic LLM fills gaps with its parametric knowledge. Even if the answer isn't in the knowledge base, it will give you something that sounds plausible.

This POC uses a **RAG + Agentic Skills** architecture to solve both problems simultaneously.

---

## 2. End-to-End Workflow

The system operates in two phases: **offline indexing** and **online Q&A**.

### 📥 Phase 1: Offline Indexing (One-Time)

```
File Upload → Document Loading → Chunking → Embedding → Vector Store
```

**Step 1 | File Upload**
Users upload PDF / TXT / MD / DOCX through Streamlit. Files are saved to the `data/` directory.

**Step 2 | Document Loading**
The appropriate loader is selected based on file extension, extracting text and metadata (filename, page number).

**Step 3 | Chunking**
`RecursiveCharacterTextSplitter` splits long documents into 1000-character chunks with 150-character overlap, preventing semantic breaks at boundaries.

**Step 4 | Embedding**
Each chunk is passed through a local ONNX model (`all-MiniLM-L6-v2`) and converted into a 384-dimensional vector. These vectors capture **semantics**, not keywords.

**Step 5 | Vector Store**
Vectors + original text + metadata are stored in ChromaDB, inserted in batches to avoid SQLite batch limits.

**Output**: A locally searchable knowledge base with zero API cost and zero network dependency.

---

### 🔍 Phase 2: Online Q&A (Per Query)

```
Question → Routing → Retrieval → Context Assembly → LLM Generation → Answer + Sources
```

**Step 6 | User Asks a Question**
For example: "What is return stacking?"

**Step 7 | Routing (The Agentic Core)**
The agent first determines what type of question this is, then selects the appropriate handling strategy. Five skills:

| Skill         | Trigger Scenario            | Output Structure                                       |
| ------------- | --------------------------- | ------------------------------------------------------ |
| 🔎 Retrieval  | Factual lookup              | Concise answer + inline citations                      |
| 📄 Summary    | Report summarization        | Executive summary + key facts + risks                  |
| 📊 Comparison | Company/strategy comparison | Neutral comparison table                               |
| ⚠️ Risk     | Risk analysis               | Market/business/financial/operational/regulatory risks |
| 🧠 Synthesis  | Cross-document synthesis    | Common themes + conflicts + open questions             |

Routing has two layers: **LLM routing** (primary, smarter) + **rule-based routing** (fallback, works even when the API fails).

**Step 8 | Semantic Retrieval**
The question is embedded and matched against the Top-K nearest chunks in ChromaDB. Smaller distance = more semantically relevant.

**Step 9 | Context Assembly**
Retrieved chunks are concatenated into a context block with source annotations — each snippet labeled with filename and page number.

**Step 10 | LLM Generation**
"System constraints + skill instruction + retrieved context" are combined into the final prompt and sent to the LLM.

The key here is the **hard constraint in the System Prompt**:

```
1. Answer ONLY based on the retrieved knowledge provided below.
2. If the retrieved knowledge does NOT contain the answer, respond EXACTLY with:
   "I don't have enough information in the knowledge base to answer this question."
3. Do NOT use your own general knowledge, do NOT infer, do NOT guess.
```

**Step 11 | Return Answer + Sources**
The answer renders in the chat interface, with an expandable Sources section so users can verify the origin of every claim.

---

## 3. Complete Data Flow

```
【OFFLINE INDEXING】
File Upload → data/
    ↓
Loaders → Documents
    ↓
TextSplitter → Chunks
    ↓
FastEmbed ONNX → 384-dim Vectors
    ↓
ChromaDB (vectors + text + metadata)

【ONLINE Q&A】
User Question
    ↓
Router (LLM / Rules) → Determines Skill
    ↓
Question Embedding → ChromaDB Similarity Search → Top-K chunks
    ↓
Assemble Context + Skill Prompt + System Prompt
    ↓
LLM Generation (grounded in context, not its own knowledge)
    ↓
Answer + Sources → Streamlit Render
```

---

## 4. Business Value

### 1️⃣ Knowledge Retrieval Upgraded from "Keywords" to "Semantics"

Traditional search can't find articles about "return stacking" when you ask "how to improve capital efficiency."
This system can — because it compares vector distances, not literal matches.

**Business value**: Researchers no longer need to remember exact wording. Natural language is enough to surface relevant knowledge.

### 2️⃣ Every Answer Is Traceable and Verifiable

Every response includes Sources. Expanding them shows the original chunk, filename, and page number.

**Business value**: Meets compliance and audit requirements in financial services. Conclusions aren't "the AI said so" — they're "according to page X of report Y."

### 3️⃣ Honest "I Don't Know" Eliminates Hallucination

Hard system prompt constraints + source attribution make the system **refuse to answer** when the knowledge base lacks coverage, instead of fabricating.

**Business value**: In finance, a fabricated number is more dangerous than no answer. This design turns "untrustworthy" into "trustworthy boundaries."

### 4️⃣ Different Questions Get Different Strategies (Agentic)

Comparison queries produce comparison tables. Summary queries produce structured summaries. Risk queries produce categorized risk lists.

**Business value**: A single entry point serves diverse analyst needs without switching tools.

### 5️⃣ Local Deployment, Zero Marginal Cost

Embeddings run on a local ONNX model. Vector storage uses local ChromaDB. Only LLM generation requires API calls.

**Business value**: Data never leaves the local environment, meeting financial data security requirements. Indexing costs nothing — API costs only occur at query time.

### 6️⃣ Extensible into a Team Knowledge Platform

The current architecture scales horizontally: add more document sources (internal research, meeting notes, regulatory filings), add more skills (valuation analysis, compliance checks), connect to enterprise vector stores.

**Business value**: Evolves from a personal tool into team knowledge infrastructure.

---

## 5. Tech Stack Summary

| Layer       | Technology                     | Role                                       |
| ----------- | ------------------------------ | ------------------------------------------ |
| Frontend    | Streamlit                      | Chat UI + file upload                      |
| Routing     | LLM + rule engine              | Determines which skill to use              |
| Retrieval   | FastEmbed (ONNX) + ChromaDB    | Semantic search                            |
| Generation  | DeepSeek / OpenAI              | Context-grounded answer generation         |
| Constraints | System Prompt                  | Prevents hallucination, enforces citations |
| Chunking    | RecursiveCharacterTextSplitter | Long-document segmentation                 |

---

## 6. One-Sentence Summary

> The core value of this POC isn't "letting an LLM answer questions" — it's **making the LLM answer only from the knowledge you provide, and tell you where every answer came from**. That's what finance actually needs from AI.
