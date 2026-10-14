# Atlas — Company Research Assistant

Northstar Systems uses Atlas to research Apple, Microsoft and Alphabet through uploaded documents and financial tables. This local teaching application connects the earlier GenAI and RAG lessons to a working prototype.

## Setup in VS Code

Extract this ZIP and open `Atlas_Company_Assistant` in VS Code. Use Python 3.11 or 3.12 (64-bit).

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Copy the environment template only if `.env` does not already exist. Edit `.env` to supply your Gemini key and an accessible generation model. The supplied model names and dependency versions are retained from the original project; account access and installation compatibility must be checked in your environment.

Run on Windows:

```powershell
.\.venv\Scripts\python.exe app.py
```

Run on macOS/Linux:

```bash
.venv/bin/python app.py
```

Open **http://127.0.0.1:7860/**. Stop the app with Ctrl+C.

## Files and workflow

- `app.py`: Gradio layout and four simple callbacks for upload, questions and clearing content, plus an error helper.
- `backend.py`: session storage, parsing, chunking, retrieval, validated pandas calculations and LangChain answer generation. Numbered comments identify the processing stages.
- `sample_data/`: the original six company files.
- `AI_Company_Assistant_Student_Guide.ipynb`: the lesson guide, including reporting-basis notes and manual evaluation cases. Read in VS Code, Jupyter or Colab; terminal commands run locally.

Upload a company’s summary and table together, supply its reporting-basis note from the guide, and repeat for the other companies. Select Documents, Tables or Both. Ask: “Compare reported FY2024 revenue for Apple, Microsoft and Alphabet, retaining each reporting-period end.”

All financial columns in the samples use USD millions. Preserve each company’s actual fiscal period. The examples are historical 2022–2024 data.

## Features retained

PDF/TXT/CSV/XLSX uploads, separate company selection, reporting notes, FAISS or Chroma, optional BM25 fusion, optional cross-encoder reranking, document citations, restricted table plans, validated pandas calculations, file inventory, evidence panels, chat display and clear buttons.

## Limits

Storage is in memory and is lost on restart. Chat history is displayed but not supplied to the model. Scanned PDFs require OCR outside the app. There is no live web lookup, forecasting, table joining, authentication, semantic cache or automated evaluation suite. Relevant passages and table samples/results are sent to Gemini. Existing upload and calculation validations are retained.

The guide records a source-data discrepancy in Microsoft’s 2023 R&D value. Original sample files are preserved; inspect source quality before drawing conclusions.

## Reading the simplified backend

Begin with `answer_question()` and follow this sequence:

| Task | Functions |
|---|---|
| Upload files | `add_files()` → `read_tables()` or `read_documents()` |
| Retrieve document evidence | `retrieve()` → `build_index()` → optional `hybrid_candidates()` → `rank_candidates()` |
| Plan table analysis | `analyze()` → `describe_tables()` → LangChain planning chain |
| Calculate results | `execute_plan()` → `validate_task()` → `apply_filters()` → `prepare_metric()` → `calculate_result()` |
| Write the answer | `generate_answer()` → LangChain answer chain |
