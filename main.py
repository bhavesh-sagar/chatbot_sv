import os
import uuid
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from pydantic import BaseModel
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI(title="AI RAG Backend (Groq Powered)")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatBody(BaseModel):
    query: str
    session_id: str | None = None

sessions = {}
chat_history = {} 

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

def extract_pdf_text(bytes_data: bytes) -> str:
    reader = PdfReader(BytesIO(bytes_data))
    text = ""
    for p in reader.pages:
        text += p.extract_text() or ""
    return text

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    text = extract_pdf_text(pdf_bytes)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    vector_store = FAISS.from_texts(chunks, emb)
    session_id = str(uuid.uuid4())
    sessions[session_id] = vector_store

    return {"session_id": session_id}

@app.post("/chat")
async def chat(body: ChatBody):
    query = body.query
    session_id = body.session_id

    # 1) Init history for this session_id (None is also a valid key)
    if session_id not in chat_history:
        chat_history[session_id] = []

    # 2) Add user message to history
    chat_history[session_id].append({"role": "user", "content": query})

    # 3) Optional RAG: only if we have a PDF vector store for this session_id
    context = ""
    if session_id is not None and session_id in sessions:
        retriever = sessions[session_id].as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        context = "\n".join(d.page_content for d in docs)

    # 4) Build conversation history text
    conv_lines: list[str] = []
    for m in chat_history[session_id]:
        conv_lines.append(f"{m['role'].title()}: {m['content']}")
    conv = "\n".join(conv_lines)

    # 5) Build final prompt for Groq
    prompt_parts = [
        "You are a helpful AI assistant.",
        "",
        "Conversation so far:",
        conv,
        "",
        "PDF Context (if available):",
        context or "(none)",
        "",
        "Based on the conversation (and PDF context if useful), answer the last user message clearly and in detail.",
    ]
    prompt = "\n".join(prompt_parts)

    # 6) Call Groq via LangChain ChatGroq
    res = llm.invoke(prompt)
    answer = res.content

    # 7) Save AI response into history
    chat_history[session_id].append({"role": "assistant", "content": answer})

    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)