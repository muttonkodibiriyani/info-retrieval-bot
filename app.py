import os
import io
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from typing import List, Tuple

import streamlit as st
import pandas as pd

# File parsers
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from PIL import Image
import pytesseract

# PDF export
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# LangChain (NEW SAFE IMPORTS)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# Config
# -----------------------------
SUPPORTED_EXTS = {
    "pdf", "docx", "xlsx", "xls", "csv", "txt", "json", "xml", "png", "jpg", "jpeg", "webp"
}

st.set_page_config(page_title="BizChat • Info Retrieval Bot", layout="wide")
st.title("BizChat • Info Retrieval Bot (Streamlit Cloud Working)")


# -----------------------------
# Utilities
# -----------------------------
def require_openai_key() -> str:
    key = st.secrets.get("OPENAI_API_KEY", None)
    if not key:
        key = os.environ.get("OPENAI_API_KEY")
    if not key:
        st.error("Missing OPENAI_API_KEY. Add it in Streamlit Cloud → Settings → Secrets.")
        st.stop()
    return key

def make_temp_dir() -> str:
    return tempfile.mkdtemp(prefix="bizchat_")

def cleanup_dir(path: str):
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

def read_pdf_bytes(file_bytes: bytes) -> str:
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            t = page.get_text("text")
            if t:
                text_parts.append(t)
    return "\n".join(text_parts).strip()

def read_docx_bytes(file_bytes: bytes) -> str:
    bio = io.BytesIO(file_bytes)
    doc = DocxDocument(bio)
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paras).strip()

def read_txt_bytes(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore").strip()
    except Exception:
        return file_bytes.decode(errors="ignore").strip()

def read_csv_bytes(file_bytes: bytes) -> str:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.to_string(index=False)

def read_excel_bytes(file_bytes: bytes) -> str:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    chunks = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        chunks.append(f"--- Sheet: {sheet} ---\n{df.to_string(index=False)}\n")
    return "\n".join(chunks).strip()

def read_json_bytes(file_bytes: bytes) -> str:
    obj = json.loads(file_bytes.decode("utf-8", errors="ignore"))
    return json.dumps(obj, indent=2, ensure_ascii=False)

def read_xml_bytes(file_bytes: bytes) -> str:
    root = ET.fromstring(file_bytes.decode("utf-8", errors="ignore"))
    lines = []

    def walk(node, depth=0):
        indent = "  " * depth
        tag = node.tag
        text = (node.text or "").strip()
        if text:
            lines.append(f"{indent}<{tag}> {text}")
        else:
            lines.append(f"{indent}<{tag}>")
        for child in list(node):
            walk(child, depth + 1)

    walk(root, 0)
    return "\n".join(lines).strip()

def read_image_ocr_bytes(file_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(file_bytes))
    txt = pytesseract.image_to_string(img)
    return (txt or "").strip()

def file_to_text(filename: str, file_bytes: bytes) -> Tuple[str, str]:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return read_pdf_bytes(file_bytes), "pdf"
    if ext == "docx":
        return read_docx_bytes(file_bytes), "docx"
    if ext in ("xlsx", "xls"):
        return read_excel_bytes(file_bytes), "excel"
    if ext == "csv":
        return read_csv_bytes(file_bytes), "csv"
    if ext == "txt":
        return read_txt_bytes(file_bytes), "txt"
    if ext == "json":
        return read_json_bytes(file_bytes), "json"
    if ext == "xml":
        return read_xml_bytes(file_bytes), "xml"
    if ext in ("png", "jpg", "jpeg", "webp"):
        return read_image_ocr_bytes(file_bytes), "image_ocr"
    return "", "unknown"

def chunk_documents(docs: List[LCDocument]) -> List[LCDocument]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def build_pdf_bytes(title: str, body: str) -> bytes:
    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=A4)
    width, height = A4

    x = 40
    y = height - 60

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title[:120])
    y -= 30

    c.setFont("Helvetica", 10)
    max_chars = 100

    for para in body.split("\n"):
        while len(para) > max_chars:
            line = para[:max_chars]
            para = para[max_chars:]
            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 60
            c.drawString(x, y, line)
            y -= 14

        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 60
        c.drawString(x, y, para)
        y -= 20

    c.save()
    buff.seek(0)
    return buff.read()

def format_sources(source_docs: List[LCDocument]) -> str:
    out = []
    for i, d in enumerate(source_docs[:6], start=1):
        src = d.metadata.get("source", "unknown")
        snippet = (d.page_content[:250] or "").replace("\n", " ").strip()
        out.append(f"[{i}] {src} — {snippet}...")
    return "\n".join(out).strip()


# -----------------------------
# Init session
# -----------------------------
api_key = require_openai_key()
os.environ["OPENAI_API_KEY"] = api_key

if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = make_temp_dir()

if "raw_docs" not in st.session_state:
    st.session_state.raw_docs = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat" not in st.session_state:
    st.session_state.chat = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Controls")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.divider()

    if st.button("🧹 Clear all data (documents + index + chat)", use_container_width=True):
        st.session_state.raw_docs = []
        st.session_state.vectorstore = None
        st.session_state.chat = []
        cleanup_dir(st.session_state.temp_dir)
        st.session_state.temp_dir = make_temp_dir()
        st.success("Cleared.")


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Upload / Index", "Chat"])

with tab1:
    colA, colB = st.columns([1.2, 1])

    with colA:
        st.subheader("1) Upload files")
        uploaded = st.file_uploader(
            "Upload PDFs, Word, Excel, CSV, JSON, XML, TXT, Images",
            type=list(SUPPORTED_EXTS),
            accept_multiple_files=True,
        )

        st.markdown("**Optional:** Paste text to include")
        pasted_text = st.text_area("Paste text here", height=160)

        if st.button("📥 Add to knowledge base", use_container_width=True):
            added, failed = 0, 0

            if pasted_text and pasted_text.strip():
                st.session_state.raw_docs.append(
                    LCDocument(page_content=pasted_text.strip(), metadata={"source": "pasted_text"})
                )
                added += 1

            if uploaded:
                for f in uploaded:
                    try:
                        data = f.read()
                        text, src_type = file_to_text(f.name, data)
                        if text and text.strip():
                            out_path = os.path.join(st.session_state.temp_dir, f.name)
                            with open(out_path, "wb") as w:
                                w.write(data)

                            st.session_state.raw_docs.append(
                                LCDocument(page_content=text, metadata={"source": f.name, "type": src_type})
                            )
                            added += 1
                        else:
                            failed += 1
                    except Exception:
                        failed += 1

            st.success(f"Added: {added} item(s). Failed/empty: {failed}.")

    with colB:
        st.subheader("2) Build / Rebuild Index (FAISS)")
        st.write(f"Documents in memory: **{len(st.session_state.raw_docs)}**")

        if st.button("⚡ Build Index", use_container_width=True, disabled=(len(st.session_state.raw_docs) == 0)):
            with st.spinner("Chunking + embedding + building index..."):
                chunks = chunk_documents(st.session_state.raw_docs)

                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

            st.success("Index built successfully ✅")

        st.divider()
        if st.session_state.vectorstore is not None:
            st.info("Index is ready. Go to **Chat** tab.")
        else:
            st.warning("No index yet. Upload docs + build index.")

        st.subheader("Current documents")
        if st.session_state.raw_docs:
            for d in st.session_state.raw_docs[:25]:
                st.write(f"- {d.metadata.get('source','unknown')} ({d.metadata.get('type','')})")
            if len(st.session_state.raw_docs) > 25:
                st.caption(f"Showing first 25 of {len(st.session_state.raw_docs)}")
        else:
            st.caption("No documents added yet.")

with tab2:
    st.subheader("Ask questions about your uploaded data")

    if st.session_state.vectorstore is None:
        st.warning("Build the FAISS index first (Upload / Index tab).")
    else:
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_q = st.chat_input("Ask a question...")
        if user_q:
            st.session_state.chat.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            llm = ChatOpenAI(model=model, temperature=temperature)
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})

            with st.spinner("Searching + answering..."):
                source_docs = retriever.get_relevant_documents(user_q)
                context = "\n\n".join(
                    [f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in source_docs]
                )

                prompt = f"""
You are a helpful assistant. Answer the user using ONLY the provided context.
If the answer is not present, say you don't know and ask what file to upload.

User question:
{user_q}

Context:
{context}

Answer:
"""
                resp = llm.invoke(prompt)
                answer = resp.content.strip()

                sources_text = format_sources(source_docs)
                final_answer = answer
                if sources_text:
                    final_answer += "\n\n---\n**Sources (snippets):**\n" + sources_text

            st.session_state.chat.append({"role": "assistant", "content": final_answer})
            with st.chat_message("assistant"):
                st.markdown(final_answer)

            pdf_bytes = build_pdf_bytes("BizChat Answer", final_answer)
            st.download_button(
                "⬇️ Download this answer as PDF",
                data=pdf_bytes,
                file_name="bizchat_answer.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
