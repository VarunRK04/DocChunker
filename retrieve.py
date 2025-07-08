import os
import json

# === CONFIG ===
UPLOADS_FOLDER = "uploads"

# === Helpers ===

def list_uploaded_docs():
    """List all document IDs available in uploads/"""
    return [f for f in os.listdir(UPLOADS_FOLDER) if os.path.isdir(os.path.join(UPLOADS_FOLDER, f))]

def get_latest_doc_id():
    """Return the most recently created upload folder"""
    folders = list_uploaded_docs()
    if not folders:
        return None
    folders.sort(key=lambda f: os.path.getctime(os.path.join(UPLOADS_FOLDER, f)), reverse=True)
    return folders[0]

def retrieve_text(doc_id):
    text_path = os.path.join(UPLOADS_FOLDER, doc_id, "text.txt")
    if not os.path.exists(text_path):
        return "⚠️ No text found."
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()

def retrieve_code(doc_id):
    code_path = os.path.join(UPLOADS_FOLDER, doc_id, "code.txt")
    if not os.path.exists(code_path):
        return "⚠️ No code found."
    with open(code_path, "r", encoding="utf-8") as f:
        return f.read()

def list_images(doc_id):
    img_folder = os.path.join(UPLOADS_FOLDER, doc_id, "images")
    if not os.path.exists(img_folder):
        return []
    return [os.path.join(img_folder, f) for f in os.listdir(img_folder)]

# === Main Program ===

# Ask which doc to load (latest by default)
doc_id = get_latest_doc_id()
if not doc_id:
    print("❌ No uploaded documents found.")
    exit(1)

print(f"\n🆔 Using latest uploaded document: {doc_id}\n")

# Ask what to retrieve
print("📂 What would you like to retrieve?")
print("1. Text\n2. Code\n3. Images")
choice = input("Enter 1 / 2 / 3: ").strip()

if choice == "1":
    text = retrieve_text(doc_id)
    print("\n📄 Text Content:\n")
    print(text[:2000] + ("\n...\n🔽 Output truncated" if len(text) > 2000 else ""))

elif choice == "2":
    code = retrieve_code(doc_id)
    print("\n💻 Code Snippets:\n")
    print(code[:2000] + ("\n...\n🔽 Output truncated" if len(code) > 2000 else ""))

elif choice == "3":
    images = list_images(doc_id)
    if not images:
        print("⚠️ No images found.")
    else:
        print("\n🖼️ Extracted Images:")
        for img in images:
            print(f"- {img}")
else:
    print("❌ Invalid choice.")
