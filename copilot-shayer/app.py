import os
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel
from PyPDF2 import PdfReader
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

load_dotenv(BASE_DIR / ".env")

SYSTEM_PROMPT = """You are an internal assistant for a law firm specialized in recuperacao judicial (judicial recovery). Your role is to support lawyers by organizing information, summarizing documents, and suggesting next steps.

Rules:

* Do NOT invent legal facts or jurisprudence
* Only use the provided document when relevant
* Be concise and structured
* If uncertain, say so clearly
* Always recommend validation by a lawyer

Output format:

* Summary (if applicable)
* Key Points
* Suggested Next Steps"""

MAX_DOCUMENT_CHARS = 18_000


class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None


class DemoDocumentStore:
    def __init__(self) -> None:
        self.clear()

    def set(self, filename: str, text: str) -> None:
        self.filename = filename
        self.text = text.strip()[:MAX_DOCUMENT_CHARS]
        self.loaded = bool(self.text)

    def clear(self) -> None:
        self.filename: Optional[str] = None
        self.text = ""
        self.loaded = False

    def status(self) -> dict:
        return {
            "loaded": self.loaded,
            "filename": self.filename,
            "characters": len(self.text),
        }


store = DemoDocumentStore()

app = FastAPI(title="Recuperacao Judicial Copilot MVP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def extract_text(upload: UploadFile, content: bytes) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix == ".txt":
        return content.decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")


def build_user_prompt(user_message: str) -> str:
    if store.loaded:
        return (
            "User question:\n"
            f"{user_message}\n\n"
            "Document content:\n"
            f"{store.text}\n\n"
            "Use the document only when relevant. If the document does not answer the question, say so clearly."
        )
    return user_message


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/api/document")
async def document_status() -> dict:
    return store.status()


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    content = await file.read()
    text = extract_text(file, content)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the file.")

    store.set(file.filename, text)
    return {
        "message": "Document loaded.",
        **store.status(),
    }


@app.delete("/api/document")
async def clear_document() -> dict:
    store.clear()
    return {"message": "Document cleared.", **store.status()}


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> dict:
    api_key = (payload.api_key or "").strip() or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Add an OpenAI API key in the app or configure OPENAI_API_KEY on the server.",
        )

    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(message)},
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {exc}") from exc

    return {"answer": response.output_text, "document": store.status()}
