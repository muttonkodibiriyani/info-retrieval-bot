import streamlit as st
import os
import tempfile
import pandas as pd
import fitz  # PyMuPDF for PDF
import pytesseract
from PIL import Image
import json
import xml.etree.ElementTree as ET
from docx import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state['documents'] = {}
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None

st.title("Universal Information Retrieval Bot")
st.write("Upload files, store data, query, analyze, and download insights.")

supported_types = ["pdf", "xlsx", "csv", "docx", "txt", "json", "xml", "jpg", "png"]

uploaded_files = st.file_uploader("Upload files", type=supported_types, accept_multiple_files=True)
text_input = st.text_area("Or paste text here")

def extract_text(file, temp_path):
    text = ""
    if file.name.endswith(".pdf"):
        doc = fitz.open(temp_path)
        for page in doc:
            text += page.get_text() + "\n"
    elif file.name.endswith(".xlsx") or file.name.endswith(".csv"):
        df = pd.read_excel(temp_path) if file.name.endswith(".xlsx") else pd.read_csv(temp_path)
        text += df.to_string() + "\n"
    elif file.name.endswith(".docx"):
        doc = Document(temp_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.name.endswith(".txt"):
        with open(temp_path, "r", encoding="utf-8") as f:
            text += f.read()
    elif file.name.endswith(".json"):
        with open(temp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            text += json.dumps(data, indent=2)
    elif file.name.endswith(".xml"):
        tree = ET.parse(temp_path)
        root = tree.getroot()
        text += ET.tostring(root, encoding="unicode")
    elif file.name.endswith((".jpg", ".png")):
        img = Image.open(temp_path)
        text += pytesseract.image_to_string(img)
    return text

if st.button("Store Data"):
    for file in uploaded_files:
        temp_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_path, "wb") as f:
            f.write(file.read())
        extracted_text = extract_text(file, temp_path)
        st.session_state['documents'][file.name] = extracted_text
    if text_input:
        st.session_state['documents'][f"text_{len(st.session_state['documents'])}"] = text_input
    st.success("Data stored successfully!")

if st.button("Build Knowledge Base"):
    if st.session_state['documents']:
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        texts = list(st.session_state['documents'].values())
        st.session_state['vectorstore'] = FAISS.from_texts(texts, embeddings)
        st.success("Knowledge base built!")
    else:
        st.error("No documents to build knowledge base.")

query = st.text_input("Ask a question about your data")
if st.button("Search") and st.session_state['vectorstore']:
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    qa = RetrievalQA.from_chain_type(llm, retriever=st.session_state['vectorstore'].as_retriever())
    answer = qa.run(query)
    st.write("### Answer:", answer)

    pdf_path = os.path.join(tempfile.gettempdir(), "answer.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(72, 720, "Query: " + query)
    c.drawString(72, 700, "Answer: " + answer)
    c.save()
    with open(pdf_path, "rb") as f:
        st.download_button("Download Answer as PDF", f, file_name="answer.pdf")

if st.button("Delete All Data"):
    st.session_state['documents'].clear()
    st.session_state['vectorstore'] = None
    st.success("All data deleted.")

st.write("### Stored Documents:")
st.write(list(st.session_state['documents'].keys()))
