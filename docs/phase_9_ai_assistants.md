# Phase 9 – AI Assistants & Natural Language UX

Use LLMs to democratize insights – let stakeholders ask questions in plain English and get answers instantly.

---
## 1  ChatGPT-like QA over Results
- Index summary CSVs & docs with **OpenAI Embeddings** → store in **PGVector**.
- FastAPI endpoint `/chat` proxies user queries, retrieves relevant chunks, and uses GPT-4o to craft answers.

Prompt skeleton:
```
You are a benefits actuarial analyst bot with access to workforce projection data …
```
Return JSON with `answer` + `citations` (file, line).

---
## 2  Jupyter AI Code-gen
Add `jupyter_ai` extension so analysts can draft pandas/plotly code snippets via `/ai` cell magic.

---
## 3  Voice Interface
Prototype voice-to-query using **OpenAI Whisper → GPT → TTS** pipeline; integrate into web UI (hold to talk button).

---
## 4  Predictive Alerting
Train light-gbm model on historical scenarios to predict *which* config changes drive audit flags (e.g. participation < 70%). Alert user before run.

---
## 5  Definition of Done
- Chat endpoint answers at ≥ 85% accuracy (manual eval set).
- Analysts generate code snippets inline.
