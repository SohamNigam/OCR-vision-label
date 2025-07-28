from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os, shutil
from typing import List

from Ocr_pipeline import (
    process_pdf,
    # ... (import your Gemini model loader here as needed)
)

gemini_model =  "gemini-2.5-pro"
conf_map =      "conf_map.json"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        temp_pdf = os.path.join("temp", f.filename)
        os.makedirs("temp", exist_ok=True)
        with open(temp_pdf, "wb") as buf:
            shutil.copyfileobj(f.file, buf)

        labeled_path, subject, date_raw = process_pdf(
            pdf_path=temp_pdf,
            gemini_model=gemini_model,
            conf_map=conf_map,
            output_folder=OUTPUT_DIR
        )
        os.remove(temp_pdf)

        results.append({
            "originalName": f.filename,
            "renamedPath": labeled_path,
            "subject": subject,
            "date": date_raw
        })
    return JSONResponse({"files": results})

@app.get("/list-output-files")
def list_output_files():
    files = []
    file_infos = []
    for fname in os.listdir(OUTPUT_DIR):
        if fname.lower().endswith(".pdf"):
            fpath = os.path.join(OUTPUT_DIR, fname)
            mtime = os.path.getmtime(fpath)
            file_infos.append((fname, mtime))
    # Sort by mtime descending (newest first)
    file_infos.sort(key=lambda x: x[1], reverse=True)
    files = [fname for fname, _ in file_infos]
    return JSONResponse({"files": files})

@app.get("/preview/{filename}")
def preview(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return Response(status_code=404)
    return StreamingResponse(open(file_path, "rb"), media_type="application/pdf")

