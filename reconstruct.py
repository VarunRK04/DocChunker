import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load credentials
load_dotenv()

# Connect to Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "text-chunks"
index = pc.Index(index_name)

# Ask for document ID
doc_id = input("📄 Enter the Document ID you want to reconstruct: ").strip()

print("🔍 Retrieving chunks from Pinecone...")

results = index.query(
    vector=[0.0] * 1536,
    top_k=1000,
    include_metadata=True,
    filter={"doc_id": {"$eq": doc_id}}
)

matches = results.get("matches", [])
if not matches:
    print(f"❌ No chunks found for doc_id: {doc_id}")
    exit(1)

# Sort by chunk_index
sorted_chunks = sorted(matches, key=lambda x: x["metadata"]["chunk_index"])

# Merge chunks into original text
reconstructed_text = "\n".join(chunk["metadata"]["text"] for chunk in sorted_chunks)

# Show preview
print("\n✅ Reconstructed Text Preview (first 1000 characters):\n")
print(reconstructed_text[:1000])

# Save full output
output_file = f"reconstructed_{doc_id}.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(reconstructed_text)

print(f"\n💾 Full reconstructed text saved to: {output_file}")
