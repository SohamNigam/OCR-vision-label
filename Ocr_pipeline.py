from google.cloud import vision
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from pdf2image import convert_from_path
import cv2
import numpy as np
import io, os, re
from datetime import datetime
from typing import List

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/stark/Downloads/OCR/vision-ocr-466709-44d985fb6a51.json"

# ---- OCR Client ----
try:
    vision_client = vision.ImageAnnotatorClient()
except Exception as e:
    print(f"Warning: Google Cloud Vision client initialization failed: {e}")
    print("Please set up Google Cloud credentials or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
    vision_client = None

BUCKET_NAME = "labeler2007"
FOLDER_PREFIX = "circuler/"  # include trailing slash to specify folder
OUTPUT_VISION_PATH = "gs://labeler2007/vision-output/"  # output folder for OCR JSON results

from google.cloud import storage

def list_pdfs_in_bucket_folder(bucket_name: str, folder_prefix: str) -> List[str]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_prefix)

    pdf_uris = []
    for blob in blobs:
        if blob.name.lower().endswith('.pdf'):
            uri = f"gs://{bucket_name}/{blob.name}"
            pdf_uris.append(uri)
    return pdf_uris


# ---- Preprocessing ----
def preprocess_image(image):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(sharpened)
    processed_image = Image.fromarray(contrast_enhanced)
    enhancer = ImageEnhance.Contrast(processed_image)
    processed_image = enhancer.enhance(1.2)
    return processed_image.convert('RGB')

# ---- PDF to Image ----
def convert_pdf_first_page(pdf_path):
    pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=300)
    if not pages:
        raise Exception("PDF conversion failed!")
    return pages[0]

# ---- OCR Extract ----
def extract_text_from_image(image):
    if vision_client is None:
        raise Exception("Google Cloud Vision client not initialized. Please set up credentials.")
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG'); img_byte_arr = img_byte_arr.getvalue()
    vision_image = vision.Image(content=img_byte_arr)
    image_context = vision.ImageContext(language_hints=['mr', 'hi', 'eng'])
    response = vision_client.document_text_detection(image=vision_image, image_context=image_context)
    if response.error.message:
        raise Exception(f'Vision API Error: {response.error.message}')
    return response.full_text_annotation.text if response.full_text_annotation else ""

# ---- Digit Correction Heuristic ----
# (Move your previously built confusion map code here)
def build_confusion_map():
    # TODO: Implement confusion map building logic
    pass
# Define conf_map globally for now, or load as an argument

def repair_token(token):
    # TODO: Implement token repair logic
    pass

def numeric_postprocess(text):
    if not text:
        return ""
    # TODO: Implement token repair logic
    return text

# ---- Subject & Date Extraction ----
import re

def extract_subject(ocr_text, gemini_model=None):
    # Try to extract line after "विषय" or "Subject"
    pattern = r"(?:विषय|Subject)\s*[:\-–—]*\s*(.+)"
    for line in ocr_text.splitlines():
        match = re.search(pattern, line, re.IGNORECASE)
        if match and match.group(1).strip():
            subject_line = match.group(1).strip()
            # Prefer 3-8 words for subject
            words = subject_line.split()
            return " ".join(words[:8])
    # If Gemini model is available, use it for subject extraction
    if gemini_model:
        prompt = "Extract a descriptive subject or title (3-8 words) for this document:"
        subject = call_gemini_model(gemini_model, prompt, ocr_text)
        return subject.strip()
    # Fallback: first non-empty line with 3+ words
    for line in ocr_text.splitlines():
        words = line.strip().split()
        if len(words) >= 3:
            return " ".join(words[:8])
    # Fallback: first non-empty line
    for line in ocr_text.splitlines():
        if line.strip():
            return " ".join(line.strip().split()[:8])
    return "No subject found"


def extract_date(text):
    if not text:
        return "No date found"
    # First, try to match after "दिनांक" or "Date"
    pattern_keyword = r"(?:दिनांक|Date)"
    date_pattern = r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})'
    pat = re.compile(pattern_keyword + r"\s*[:\-]*\s*" + date_pattern, re.IGNORECASE)
    for line in text.splitlines():
        match = pat.search(line)
        if match:
            return match.group(1)
    # Fallback: match any date-like pattern anywhere in the text
    generic_date_pat = re.compile(r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})')
    match = generic_date_pat.search(text)
    if match:
        return match.group(1)
    return "No date found"

def convert_devanagari_to_english(text):
    if text is None:
        return ""
    devanagari_digits = "०१२३४५६७८९"
    eng_digits = "0123456789"
    return text.translate(str.maketrans(devanagari_digits, eng_digits))

def clean_filename(text):
    if text is None:
        text = ""
    else:
        text = str(text)
    text = re.sub(r"[\/\.\-]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")[:100]

# ---- Summarize subject ----
def summarize_with_gemini(text, gemini_model):
    # Simple placeholder: just return the input text
    if not text:
        return "No summary found"
    return text

def summarize_subject(text, gemini_model=None, max_words=4):
    # Use Gemini or fallback to first non-empty line
    if gemini_model:
        prompt = f"Extract a concise subject or subtitle ({max_words} words max) summarizing the main topic of this document:"
        summary = call_gemini_model(gemini_model, prompt, text)
    else:
        # Fallback: use first non-empty line
        lines = text.splitlines()
        summary = ""
        for line in lines:
            if line.strip():
                summary = line.strip()
                break
    # Trim summary to max_words
    words = summary.strip().split()
    if len(words) > max_words:
        summary = ' '.join(words[:max_words])
    return summary or "No subject found"

# ---- Gemini API Call ----
def call_gemini_model(model, prompt, text):
    # TODO: Implement Gemini API call here
    # For now, just return the first line of text
    lines = text.splitlines()
    for line in lines:
        if line.strip():
            return line.strip()
    return "No subject found"

# ---- MAIN PIPELINE FUNCTION ----
import re

def process_pdf(pdf_path, gemini_model, conf_map, output_folder=None):
    image = convert_pdf_first_page(pdf_path)
    processed = preprocess_image(image)
    raw_text = extract_text_from_image(processed)
    ocr_text = numeric_postprocess(raw_text)
    subject = extract_subject(ocr_text, gemini_model=gemini_model)
    date_raw = extract_date(ocr_text) or "No date found"
    date_eng = convert_devanagari_to_english(date_raw)
    subject_clean = clean_filename(subject)
    date_clean = clean_filename(date_eng)
    final_filename = f"{subject_clean}_{date_clean}.pdf"
    if output_folder:
        import shutil
        new_path = os.path.join(output_folder, final_filename)
        shutil.copy2(pdf_path, new_path)
        return new_path, subject, date_raw
    return final_filename, subject, date_raw

def run_batch_vision_ocr(pdf_gs_uris: List[str], output_uri_prefix: str):
    client = vision.ImageAnnotatorClient()
    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

    requests = []
    for uri in pdf_gs_uris:
        gcs_source = vision.GcsSource(uri=uri)
        input_config = vision.InputConfig(gcs_source=gcs_source, mime_type="application/pdf")
        gcs_dest = vision.GcsDestination(uri=output_uri_prefix)
        output_config = vision.OutputConfig(gcs_destination=gcs_dest, batch_size=10)  # tune batch_size if needed

        request = vision.AsyncAnnotateFileRequest(
            features=[feature],
            input_config=input_config,
            output_config=output_config,
        )
        requests.append(request)

    operation = client.async_batch_annotate_files(requests=requests)
    print("Waiting for OCR batch operation to complete...")
    operation.result(timeout=1800)  # wait max 30 min, adjust if needed
    print("Batch OCR completed.")
    return operation
