# Company Knowledge and Comparison Assistant

A simplified teaching project for VS Code. Exactly two application Python files:
`backend.py` (processing) and `app.py` (Gradio + FastAPI).

## Folder contents

- `backend.py`: LangChain prompts/chains, document ingestion, recursive chunking, local embeddings, FAISS/Chroma, BM25 fusion, cross-encoder reranking, validated pandas plans, cited answers.
- `app.py`: UI controls, session state, error messages, API routes.
- `Company_Assistant_Guide.ipynb`: three-section teaching notebook with exercises and expected results.
- `requirements.txt`: verified top-level package versions; Gradio is intentionally kept on version 5.
- `.env.example`: configuration template; copy it to `.env`.
- `sample_data/`: synthetic company reports, CSVs, XLSX, sample questions and answers.

## Windows / VS Code setup

1. Extract the ZIP and open the `Company_Assistant_LangChain` folder in VS Code.
2. Use Python 3.11 or 3.12 and run these commands in a terminal opened in this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

3. Open `.env` and set your Gemini API key. Keep a model ID supported by your account;
   change `GEMINI_MODEL` if the example model is unavailable. Do not put quotes around the key.
4. In VS Code install Microsoft's Python and Jupyter extensions. Select `.venv` as the
   Python interpreter and notebook kernel.
5. Start the app:

```powershell
.\.venv\Scripts\python.exe app.py
```

6. Open http://127.0.0.1:8000/ui/ . API docs: http://127.0.0.1:8000/docs .
   Stop the server with Ctrl+C.

macOS/Linux: use `python3 -m venv .venv` and `.venv/bin/python` for the commands above;
copy the template with `cp .env.example .env`.

## First demonstration

Upload files **one company at a time**:

| Company label | Upload | Reporting basis |
| --- | --- | --- |
| Company Alpha | `alpha_sales.csv`, `alpha_report.txt` | INR actual units; Jan–Mar 2026; one row per month; revenue and profit are disjoint monthly amounts. |
| Company Beta | `beta_sales.csv`, `beta_report.txt` | INR actual units; Jan–Mar 2026; one row per month; revenue and profit are disjoint monthly amounts. |

Ask: "Compare Q1 2026 total revenue with each company's documented Q1 target."
Select both companies and **Both** as the evidence source.
Expected: Alpha revenue 45,000 versus target 50,000; Beta revenue 54,000 versus target
50,000. The answer must cite table evidence for actual revenue and document evidence
for targets. Evidence IDs are generated per question and can vary.

Use FAISS first. Switch to Chroma and repeat a document question. Toggle reranking
or keyword search and inspect the evidence panel. A changed rank does not guarantee
an improved answer; evaluate against the question and source text.

Do not upload both `alpha_sales.csv` and the equivalent `alpha_sales.xlsx` to represent
additional sales: they contain duplicate data. The XLSX is an alternative format demo.

## Architecture

Document branch: load -> LangChain Documents -> recursive chunks -> normalized embeddings
-> selected vector store -> dense retrieval (optionally fused with BM25) -> optional
cross-encoder reranking -> source-labelled context -> LangChain answer chain.

Table branch: CSV/XLSX -> DataFrames -> schema/sample descriptions -> LangChain structured
plan -> Pydantic and execution validation -> predefined pandas calculations -> cited answer.

Both: combine document evidence and calculated table evidence in one answer prompt.
Questions are standalone; chat history is shown in the UI but is not supplied to Gemini.

## Scope and limitations

- Default document search returns up to six candidates and three final passages per company.
  FAISS and Chroma use the same embeddings and chunks but may return different rankings.
- Chunk size/overlap are **characters**, not tokens. This is recursive chunking, not semantic chunking.
- Reranking scores are relevance scores, not confidence probabilities. No automatic relevance
  threshold is claimed; unrelated passages must not be forced into answers.
- Embeddings and reranking run locally on CPU after the first model download. Model downloads
  can be large. Schema samples, selected document passages, and calculation results are sent to Gemini.
  API access/quota is required only for planning and answer generation, not manual pandas calculations.
- Main-app indexes and workspaces are in memory. Server restart loses uploaded session data.
  The notebook separately demonstrates safe FAISS index persistence and Chroma persistence/CRUD;
  it does not implement automatic app-session restoration.
- Tables support sum/mean/min/max/count, up to two grouping columns and five filters per task.
  No joins, arbitrary code, first/last closing prices, daily returns, moving averages, or forecasting.
- Separate monthly CSVs for one company are accepted under the same label. The planner can
  retrieve independent totals from disjoint files; one combined chronological table needs an
  explicit concatenation step. Specify non-overlapping periods in the basis notes.
- Numeric cells must be plain numbers with consistent units. Unsupported calculations should
  be reported as unavailable rather than guessed.
- PDFs need extractable text. No OCR is included. Limits: 20 files/workspace, 5 MB/file,
  100 PDF pages, 10 XLSX sheets, 20,000 rows/table, 500 chunks/company, five selected companies.
- Prompt separation and validation reduce some risks; prompts alone do not guarantee injection resistance.
  This local teaching app has no authentication or production multi-user deployment setup.
- Hybrid search is implemented using BM25 plus reciprocal rank fusion (RRF). Semantic caching,
  semantic chunking, advanced FAISS indexes, streaming, and Ragas are extension exercises,
  not implemented production features.



## Primary documentation

- https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai
- https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma
- https://docs.langchain.com/oss/python/integrations/vectorstores/faiss
- https://sbert.net/docs/cross_encoder/usage/usage.html
- https://github.com/facebookresearch/faiss/wiki
- https://docs.trychroma.com/
- https://docs.ragas.io/en/stable/

See `VERIFICATION.md` for checks completed in the supplied execution environment.
