import os
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from utils import extract_text_from_pdf, chunk_text

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

# Use your existing index
index_name = "test-index"
vector_dim = 1024  # ✅ Matches your Pinecone index

# Connect to the index (skip creation since it already exists)
index = pc.Index(index_name)

# Path to your PDF
pdf_path = "D:/project_3_pinecone/petpulsereport.pdf"
if not os.path.exists(pdf_path):
    print(f"❌ File not found: {pdf_path}")
    exit(1)

# Extract + chunk
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)

# ✅ Safe fake embedding (consistent, non-zero values)
def fake_embedding(text):
    vec = [((ord(c) % 100) / 100) for c in text[:vector_dim]]
    return vec + [0.0] * (vector_dim - len(vec))

# Uploading
doc_id = str(uuid.uuid4())
vectors = []

for i, chunk in enumerate(chunks):
    if not chunk.strip():
        continue  # Skip empty
    embedding = fake_embedding(chunk)
    if all(val == 0.0 for val in embedding):
        continue  # Skip all-zero
    vectors.append({
        "id": f"{doc_id}-{i}",
        "values": embedding,
        "metadata": {
            "doc_id": doc_id,
            "chunk_index": i,
            "text": chunk
        }
    })

if vectors:
    response = index.upsert(vectors)
    print(f"✅ Uploaded {len(vectors)} chunks to index '{index_name}'")
    print("🔁 Pinecone response:", response)
    print(f"🆔 Document ID: {doc_id}")
else:
    print("⚠️ No valid chunks to upload.")
