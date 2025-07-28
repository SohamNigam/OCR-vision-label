from Ocr_pipeline import list_pdfs_in_bucket_folder, run_batch_vision_ocr, BUCKET_NAME, FOLDER_PREFIX, OUTPUT_VISION_PATH

def main():
    pdf_uris = list_pdfs_in_bucket_folder(BUCKET_NAME, FOLDER_PREFIX)
    run_batch_vision_ocr(pdf_uris, OUTPUT_VISION_PATH)
    print("Batch OCR submitted and completed.")

if __name__ == "__main__":
    main()
