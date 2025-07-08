import os
import uuid
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Load .env (optional if using Pinecone later)
load_dotenv()

# === CONFIGURATION ===
pdf_path = "D:/project_3_pinecone/petpulsereport.pdf"
base_output_dir = "uploads"
os.makedirs(base_output_dir, exist_ok=True)

# === HELPERS ===

def extract_text_and_code(text):
    """Separate text and code using simple keyword-based heuristics."""
    lines = text.splitlines()
    text_lines = []
    code_lines = []
    in_code = False

    for line in lines:
        # Detect code blocks
        if any(line.strip().startswith(kw) for kw in ("import", "def", "class", "void", "main(", "if", "for", "while")) or \
           any(sym in line for sym in ("{", "}", "=>", ";", "//")):
            in_code = True
            code_lines.append(line)
        elif in_code and line.strip() == "":
            in_code = False
            text_lines.append("")  # blank line
        elif in_code:
            code_lines.append(line)
        else:
            text_lines.append(line)

    return "\n".join(text_lines).strip(), "\n".join(code_lines).strip()

def extract_images(pdf_path, image_output_dir):
    """Extract and save images from a PDF."""
    doc = fitz.open(pdf_path)
    count = 0
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            image_path = os.path.join(image_output_dir, f"image_{page_index + 1}_{img_index + 1}.{ext}")
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            count += 1
    return count

# === MAIN PROCESS ===

# Generate unique folder for this upload
doc_id = str(uuid.uuid4())
upload_dir = os.path.join(base_output_dir, doc_id)
os.makedirs(upload_dir, exist_ok=True)

# Create subfolder for images
image_folder = os.path.join(upload_dir, "images")
os.makedirs(image_folder, exist_ok=True)

# Load PDF
doc = fitz.open(pdf_path)

all_text = []
for page in doc:
    all_text.append(page.get_text())

full_text = "\n".join(all_text)

# Separate text and code
plain_text, code_snippets = extract_text_and_code(full_text)

# Save text
with open(os.path.join(upload_dir, "text.txt"), "w", encoding="utf-8") as f:
    f.write(plain_text)

# Save code
with open(os.path.join(upload_dir, "code.txt"), "w", encoding="utf-8") as f:
    f.write(code_snippets)

# Save images
image_count = extract_images(pdf_path, image_folder)

# Summary
print(f"✅ Upload complete")
print(f"📁 Stored in: uploads/{doc_id}/")
print(f"📝 Text lines: {len(plain_text.splitlines())}")
print(f"💻 Code lines: {len(code_snippets.splitlines())}")
print(f"🖼️ Images extracted: {image_count}")
print(f"🆔 Document ID: {doc_id}")
