import logging
import tempfile

import requests
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from services.file_handler import create_output_dir, extract_zip, save_uploaded_file
from services.pdf_processor import process_pdfs, save_images_as_gray

# Setup logging
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = FastAPI()

OCR4ALL_DIR_URL = "http://localhost:1476/ocr4all/ajax/overview/checkDir"
OCR4ALL_PAGE_URL = "http://localhost:1476/ocr4all/ajax/generic/pagelist"
OCR4ALL_EXECUTE_URL = "http://localhost:1476/ocr4all/ajax/processFlow/execute"

HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Origin": "http://localhost:1476",
    "Referer": "http://localhost:1476/ocr4all/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0",
    "X-Requested-With": "XMLHttpRequest",
    "sec-ch-ua": '"Opera GX";v="109", "Not:A-Brand";v="8", "Chromium";v="123"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}


def fetch_cookies():
    session = requests.Session()
    return session.cookies


@app.post("/upload-zip/")
async def upload_zip(file: UploadFile = File(...)):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = save_uploaded_file(file, tmpdir)
            extract_zip(zip_path, tmpdir)

            pdf_paths = process_pdfs(tmpdir)
            for pdf_path in pdf_paths:
                output_dir = create_output_dir(pdf_path)
                save_images_as_gray(pdf_path, output_dir)

            logging.info(f"Processed all PDFs in {file.filename} successfully.")
    except Exception as e:
        logging.error(f"Error processing zip file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing zip file: {e}")

    return JSONResponse(
        {
            "status": "success",
            "message": "Files processed and copied to OCR4all input directory.",
        }
    )


@app.get("/get_files_dir/")
async def get_files_dir(
    projectDir: str,
    imageType: str = "Binary",
    resetSession: bool = True,
    cookies: dict = Depends(fetch_cookies),
):
    try:
        params = {
            "projectDir": "%2Fvar%2Focr4all%2Fdata%2F" + projectDir + "%2F",
            "imageType": imageType,
            "resetSession": resetSession,
        }
        response = await requests.get(
            OCR4ALL_DIR_URL, params=params, cookies=cookies, headers=HEADERS
        )
        response.raise_for_status()
        return response.json(), response.cookies
    except requests.RequestException as e:
        logging.error(f"Error fetching files from OCR4all: {e}")
        raise HTTPException(status_code=500, detail="Error fetching files from OCR4all")


@app.get("/get_pages/")
async def get_pages(
    imageType: str = "Original", cookies: dict = Depends(fetch_cookies)
):
    try:
        params = {"imageType": imageType}
        response = requests.get(
            OCR4ALL_PAGE_URL, params=params, cookies=cookies, headers=HEADERS
        )
        response.raise_for_status()
        return response.json(), response.cookies
    except requests.RequestException as e:
        logging.error(f"Error fetching pages from OCR4all: {e}")
        raise HTTPException(status_code=500, detail="Error fetching pages from OCR4all")


@app.post("/start_processing/")
async def start_processing(
    request: Request, payload: dict, cookies: dict = Depends(fetch_cookies)
):
    try:
        response = requests.post(
            OCR4ALL_EXECUTE_URL, json=payload, cookies=cookies, headers=HEADERS
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error starting processing in OCR4all: {e}")
        raise HTTPException(
            status_code=500, detail="Error starting processing in OCR4all"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
