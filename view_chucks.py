import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load .env variables
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index = pc.Index("text-chunks")

# Ask for document ID
doc_id = input("🆔 Enter the Document ID: ").strip()

# Query chunks using metadata filter
results = index.query(
    vector=[0.0] * 1536,  # dummy vector
    top_k=1000,
    include_metadata=True,
    filter={"doc_id": {"$eq": doc_id}}
)

matches = results.get("matches", [])

# Show each chunk
if matches:
    print(f"\n📄 Found {len(matches)} chunks for document {doc_id}:\n")
    for chunk in sorted(matches, key=lambda x: x['metadata']['chunk_index']):
        print(f"[{chunk['metadata']['chunk_index']}] → {chunk['metadata']['text'][:200]}...\n")
else:
    print("❌ No chunks found.")
