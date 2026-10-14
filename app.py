"""Gradio interface and FastAPI routes. All processing lives in backend.py."""
import logging
import os
from pathlib import Path
from typing import Literal

# Allow local browser/API requests to bypass any configured proxy.
for name in ("NO_PROXY", "no_proxy"):
    os.environ[name] = ",".join(filter(None, [os.getenv(name, ""), "localhost", "127.0.0.1", "::1"]))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
import backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("company_assistant")
api = FastAPI(title="Atlas — Company Research Assistant", version="3.0")


class AskRequest(BaseModel):
    # Pydantic checks API request fields before our route executes.
    workspace_id: str
    question: str = Field(min_length=1, max_length=2000)
    companies: list[str] = Field(min_length=1, max_length=5)
    source: Literal["Documents", "Tables", "Both"] = "Both"
    store_type: Literal["FAISS", "Chroma"] = "FAISS"
    rerank: bool = True
    hybrid: bool = False


def api_error(exc):
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    logger.warning("Backend failed (%s)", type(exc).__name__)
    return HTTPException(status_code=503, detail="Processing failed. Check model access, downloads, network, and quota.")


@api.get("/")
def home():
    return RedirectResponse("/ui/")


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/workspaces")
def new_workspace():
    try:
        return {"workspace_id": backend.create_workspace()}
    except Exception as exc:
        raise api_error(exc) from None


@api.get("/workspaces/{workspace_id}")
def list_files(workspace_id: str):
    try:
        companies, files = backend.inventory(workspace_id)
        return {"companies": companies, "inventory": files}
    except Exception as exc:
        raise api_error(exc) from None


@api.delete("/workspaces/{workspace_id}")
def clear_files(workspace_id: str):
    try:
        return {"message": backend.clear_workspace(workspace_id)}
    except Exception as exc:
        raise api_error(exc) from None


@api.post("/workspaces/{workspace_id}/upload")
async def upload_files(workspace_id: str, company: str = Form(...), basis: str = Form(""),
                       files: list[UploadFile] = File(...)):
    try:
        if len(files) > 20:
            raise ValueError("Upload at most 20 files.")
        payload = [(f.filename or "upload", await f.read(backend.MAX_BYTES + 1)) for f in files]
        # File parsing is synchronous; run it in a thread rather than blocking the event loop.
        message = await run_in_threadpool(backend.add_files, workspace_id, company, payload, basis)
        return {"message": message}
    except Exception as exc:
        raise api_error(exc) from None
    finally:
        for f in files:
            await f.close()


@api.post("/ask")
def ask_api(request: AskRequest):
    try:
        return backend.answer_question(request.workspace_id, request.question, request.companies,
                                       request.source, request.store_type, request.rerank, request.hybrid)
    except Exception as exc:
        raise api_error(exc) from None


def ui_error(exc):
    """Show the actual error so we can diagnose failed requests."""
    if isinstance(exc, ValueError):
        return gr.Error(str(exc))

    # Print the original backend traceback and Gemini error in the terminal.
    logger.exception("UI backend failed")

    # Display the provider's error message in the interface.
    return gr.Error(f"{type(exc).__name__}: {exc}")

def add_ui(key, company, files, basis):
    try:
        if not files:
            raise ValueError("Select files first.")
        payload = []
        for file in files:
            path = Path(file)
            if path.stat().st_size > backend.MAX_BYTES:
                raise ValueError("Each file must be at most 5 MB.")
            payload.append((path.name, path.read_bytes()))
        message = backend.add_files(key, company or "", payload, basis or "")
        companies, rows = backend.inventory(key)
        return message, gr.update(choices=companies, value=companies[:5]), rows, [], None, None
    except Exception as exc:
        raise ui_error(exc) from None


def ask_ui(key, question, companies, source, store_type, rerank, hybrid, history):
    try:
        result = backend.answer_question(key, question or "", companies or [], source, store_type, rerank, hybrid)
        # History is for display only. Each backend question is standalone.
        chat = list(history or []) + [{"role": "user", "content": question},
                                      {"role": "assistant", "content": result["answer"]}]
        return "", chat, result["documents"], result["calculations"]
    except Exception as exc:
        raise ui_error(exc) from None


def reset_ui(key):
    try:
        backend.clear_workspace(key)
        return "Workspace cleared.", gr.update(choices=[], value=[]), [], [], None, None, None
    except Exception as exc:
        raise ui_error(exc) from None


CSS = """
.gradio-container {max-width:1400px !important; margin:auto;}
#hero {padding:24px; border-radius:16px; background:#172554; margin-bottom:18px;}
#hero h1 {color:white; margin:0 0 10px;}
#hero p {color:#dbeafe; margin:0;}
"""
with gr.Blocks(title="Atlas by Northstar Systems", css=CSS,
               theme=gr.themes.Soft(primary_hue="blue")) as ui:
    gr.HTML('<div id="hero"><h1>Atlas — Company Research Assistant</h1>'
            '<p>Northstar Systems’ research workspace for comparing company documents and financial tables with supporting evidence.</p></div>')
    key = gr.State("")
    ui.load(backend.create_workspace, inputs=None, outputs=key)
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 1. Upload company files")
            company = gr.Textbox(label="Company label", placeholder="Apple, Microsoft or Alphabet")
            files = gr.File(label="PDF / TXT / CSV / XLSX", file_count="multiple",
                            file_types=[".pdf", ".txt", ".csv", ".xlsx"], type="filepath")
            basis = gr.Textbox(label="Reporting basis (required for tables)", lines=3,
                              placeholder="INR actual units; Jan–Mar 2026; one row per month; disjoint monthly records.")
            add = gr.Button("Add company files", variant="primary")
            status = gr.Textbox(label="Status", value="Upload one company at a time.", interactive=False)
            selected = gr.Dropdown([], multiselect=True, label="Companies to analyze", interactive=True)
            source = gr.Radio(["Documents", "Tables", "Both"], value="Both", label="Evidence source")
            with gr.Accordion("Retrieval settings (document questions)", open=True):
                store_type = gr.Radio(["FAISS", "Chroma"], value="FAISS", label="Vector store")
                rerank = gr.Checkbox(value=True, label="Rerank retrieved passages")
                hybrid = gr.Checkbox(value=False, label="Include keyword search (BM25)")
            reset = gr.Button("Clear uploaded files")
            gr.Markdown("Session files and indexes are cleared when the server restarts. "
                        "Relevant excerpts and table samples/results are sent to Gemini.")
        with gr.Column(scale=3, min_width=320):
            gr.Markdown("### 2. Ask and compare")
            chat = gr.Chatbot(type="messages", height=380, show_label=False)
            question = gr.Textbox(label="Question", lines=2,
                                  placeholder="Compare reported FY2024 revenue for Apple, Microsoft and Alphabet, retaining each reporting-period end.")
            with gr.Row():
                ask = gr.Button("Ask Atlas", variant="primary")
                clear = gr.Button("Clear conversation")
            with gr.Accordion("Uploaded files", open=True):
                inventory = gr.Dataframe(headers=["Company", "File", "Sheet/type", "Rows", "Columns", "Table ID"],
                                         interactive=False)
            with gr.Accordion("Inspect evidence", open=False):
                doc_output = gr.JSON(label="Document excerpts, sources, and reranking scores")
                calc_output = gr.JSON(label="Verified table calculations")
            gr.Markdown("Each question is standalone. State company names and the period explicitly.")
    options = {"concurrency_limit": 1, "concurrency_id": "workspace"}
    add.click(add_ui, [key, company, files, basis],
              [status, selected, inventory, chat, doc_output, calc_output], **options)
    for trigger in (ask.click, question.submit):
        trigger(ask_ui, [key, question, selected, source, store_type, rerank, hybrid, chat],
                [question, chat, doc_output, calc_output], **options)
    clear.click(lambda: ("", [], None, None), None, [question, chat, doc_output, calc_output], **options)
    reset.click(reset_ui, [key], [status, selected, inventory, chat, doc_output, calc_output, files], **options)

app = gr.mount_gradio_app(api, ui.queue(), path="/ui")
if __name__ == "__main__":
    print("UI: http://127.0.0.1:8000/ui/ | API docs: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
