import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    """
    Extract text from all pages of a PDF file using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """
    Split the input text into manageable chunks using LangChain's splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
