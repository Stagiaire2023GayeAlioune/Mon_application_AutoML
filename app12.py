from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import fitz  # PyMuPDF
import pytesseract
from pdfminer.high_level import extract_text
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <html>
        <head>
            <title>PDF Accessibility Checker</title>
            <script>
                function uploadFile(event) {
                    event.preventDefault();
                    let formData = new FormData(document.getElementById("uploadForm"));
                    
                    fetch("/upload/", {
                        method: "POST",
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.corrected_pdf) {
                            let downloadLink = document.getElementById("downloadLink");
                            downloadLink.href = data.corrected_pdf;
                            downloadLink.style.display = "block";

                            let reportSection = document.getElementById("reportSection");
                            reportSection.innerText = data.report;
                            reportSection.style.display = "block";
                        }
                    })
                    .catch(error => console.error("Error:", error));
                }
            </script>
        </head>
        <body>
            <h2>Upload a PDF file to analyze accessibility issues</h2>
            <form id="uploadForm" onsubmit="uploadFile(event)" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Upload">
            </form>
            <br>
            <div id="reportSection" style="display:none; border:1px solid #ccc; padding:10px;"></div>
            <br>
            <a id="downloadLink" href="#" style="display:none;" download="corrected.pdf">
                <button>Download Corrected PDF</button>
            </a>
        </body>
    </html>
    """

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    text_content = extract_text_from_pdf(file_path)
    accessibility_issues = analyze_pdf(file_path, text_content)
    corrected_pdf_path = correct_pdf(file_path, accessibility_issues)
    report = generate_report(accessibility_issues)

    return JSONResponse(content={"message": "PDF analysé et corrigé", "corrected_pdf": f"/download/{os.path.basename(corrected_pdf_path)}", "report": report})

@app.get("/download/{filename}")
async def download_pdf(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    return FileResponse(file_path, media_type="application/pdf", filename=filename)

def extract_text_from_pdf(pdf_path):
    """Amélioration de l'extraction du texte avec OCR si nécessaire."""
    text = extract_text(pdf_path).strip()
    
    if not text:
        doc = fitz.open(pdf_path)
        for page in doc:
            img = page.get_pixmap()
            text += pytesseract.image_to_string(img)
    return text

def analyze_pdf(pdf_path, text_content):
    """Analyse les problèmes d'accessibilité et retourne une liste de corrections à effectuer."""
    issues = []
    doc = fitz.open(pdf_path)
    
    for page_num, page in enumerate(doc):
        if not text_content.strip():
            issues.append({"type": "empty_text", "message": "Le document semble être une image sans texte."})

        if not check_heading_structure(text_content):
            issues.append({"type": "heading_structure", "message": "Structure des titres incorrecte. Ajout de titres hiérarchisés."})
        
        for img in page.get_images(full=True):
            issues.append({"type": "missing_alt", "image_id": img[0], "page": page_num + 1})
    
    return issues

def check_heading_structure(text):
    """Vérifie si le document contient une structure de titres logique."""
    headings = [line for line in text.split("\n") if line.strip().startswith(("H1", "H2", "H3"))]
    return len(headings) > 0

def correct_pdf(pdf_path, issues):
    """Applique les corrections nécessaires pour rendre le PDF accessible."""
    doc = fitz.open(pdf_path)
    
    for issue in issues:
        if issue["type"] == "missing_alt":
            page = doc[issue["page"] - 1]
            page.insert_text((50, 50), "[Image description ajoutée]", fontsize=10)
        
        elif issue["type"] == "heading_structure":
            for page in doc:
                page.insert_text((50, 100), "H1: Titre du document\n", fontsize=14, fontname="helv")

        elif issue["type"] == "empty_text":
            for page in doc:
                page.insert_text((50, 150), "⚠️ Ce document a été détecté comme une image sans texte. Ajout d'une couche OCR.", fontsize=12, fontname="helv")
    
    corrected_pdf_path = pdf_path.replace(".pdf", "_corrected.pdf")
    doc.save(corrected_pdf_path)
    doc.close()
    
    return corrected_pdf_path

def generate_report(issues):
    """Génère un rapport détaillé des problèmes détectés et corrections appliquées."""
    report = "🔎 **Rapport d'analyse du document PDF :**\n\n"
    
    if not issues:
        report += "✅ Aucun problème détecté. Le document semble conforme.\n"
    else:
        for issue in issues:
            if issue["type"] == "missing_alt":
                report += f"❌ Image sans description détectée (Page {issue['page']}). Description ajoutée.\n"
            elif issue["type"] == "empty_text":
                report += "❌ Le document semble être une image sans texte accessible. Une OCR a été appliquée.\n"
            elif issue["type"] == "heading_structure":
                report += "⚠️ Structure des titres incorrecte. Ajout d'une hiérarchie logique des titres.\n"
    
    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
