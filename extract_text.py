# ======= SAFETENSORS PATCH (must come before importing paddleocr) =======
import importlib
from safetensors import safe_open

def safe_open_pt(path, framework="paddle", *args, **kwargs):
    """Force safetensors to use PyTorch backend instead of unsupported Paddle."""
    if framework == "paddle":
        framework = "pt"
    return safe_open(path, framework=framework, *args, **kwargs)

# Patch every module that might import safe_open
def apply_safetensors_patch():
    for name in list(importlib.sys.modules):
        if name and "paddlex" in name and "model_utils" in name:
            try:
                mod = importlib.import_module(name)
                setattr(mod, "safe_open", safe_open_pt)
            except Exception:
                pass
apply_safetensors_patch()
# ======= END PATCH =======================================================


from paddleocr import PaddleOCRVL
import fitz
import os
from PIL import Image


# === 1. Load the PDF ===
pdf_path = "TDF Cricket - 2025 Rule Book_v2.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

doc = fitz.open(pdf_path)
n_pages = doc.page_count
print(f"✅ PDF loaded: {pdf_path}")
print(f"📄 Number of pages: {n_pages}")

# === 2. Load PaddleOCR-VL model ===
print("⏳ Loading PaddleOCR-VL model... (this may take a few minutes)")

import paddlex.inference.models.common.vlm.transformers.model_utils as mu
from safetensors import safe_open
def safe_open_pt(path, framework="paddle", *args, **kwargs):
    if framework == "paddle":
        framework = "pt"
    return safe_open(path, framework=framework, *args, **kwargs)
mu.safe_open = safe_open_pt

ocr = PaddleOCRVL()  # automatically downloads model files

# === 3. Convert PDF pages to images and extract text ===
output_dir = "ocr_output"
os.makedirs(output_dir, exist_ok=True)

for i in range(n_pages):
    page = doc.load_page(i)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_path = os.path.join(output_dir, f"page_{i+1:02d}.png")
    pix.save(img_path)
    print(f"\n🔍 Processing page {i+1}/{n_pages}...")

    # Run OCR
    results = ocr.predict(img_path)

    # Save output for each page
    for j, res in enumerate(results):
        json_path = os.path.join(output_dir, f"page_{i+1:02d}_res_{j}.json")
        md_path = os.path.join(output_dir, f"page_{i+1:02d}_res_{j}.md")
        res.save_to_json(json_path)
        res.save_to_markdown(md_path)
        print(f"✅ Saved page {i+1} output to {json_path} and {md_path}")

print("\n🎉 Extraction complete!")
print(f"All results are saved in the '{output_dir}' folder.")
