from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os, shutil
from Ocr_pipeline import process_pdf
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # ← closing parenthesis below
)

# Create OUTPUT_DIR if it doesn't exist
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount 'output' as static files
app.mount(
    "/output",
    StaticFiles(directory=os.path.join(os.getcwd(), OUTPUT_DIR)),
    name="output"
)

# ... your upload, list, and preview endpoints follow ...


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        temp_pdf = os.path.join("temp", f.filename)
        os.makedirs("temp", exist_ok=True)
        
        with open(temp_pdf, "wb") as buf:
            shutil.copyfileobj(f.file, buf)
        
        try:
            # This is the key fix - properly calling process_pdf with output_folder
            labeled_path, subject, date_raw = process_pdf(
                pdf_path=temp_pdf,
                gemini_model=None,
                conf_map=None,
                output_folder=OUTPUT_DIR  # This ensures files are saved to output folder
            )
            
            results.append({
                "originalName": f.filename,
                "renamedPath": labeled_path,
                "subject": subject,
                "date": date_raw
            })
            
        except Exception as e:
            print(f"Error processing {f.filename}: {e}")
            results.append({
                "originalName": f.filename,
                "error": str(e)
            })
        
        # Clean up temp file
        if os.path.exists(temp_pdf):
            os.remove(temp_pdf)
    
    return JSONResponse({"files": results})

@app.get("/list-output-files")
def list_output_files():
    try:
        files = []
        file_infos = []
        for fname in os.listdir(OUTPUT_DIR):
            if fname.lower().endswith(".pdf"):
                fpath = os.path.join(OUTPUT_DIR, fname)
                mtime = os.path.getmtime(fpath)
                file_infos.append((fname, mtime))
        
        file_infos.sort(key=lambda x: x[1], reverse=True)
        files = [fname for fname, _ in file_infos]
        return JSONResponse({"files": files})
    except Exception as e:
        print(f"Error listing files: {e}")
        return JSONResponse({"files": []})

@app.get("/preview/{filename}")
def preview(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return Response(status_code=404, content="File not found")
    return StreamingResponse(open(file_path, "rb"), media_type="application/pdf")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
