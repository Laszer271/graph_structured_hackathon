import os
import shutil
from zipfile import ZipFile
from fastapi import UploadFile # type: ignore

def save_uploaded_file(file: UploadFile, destination_dir: str) -> str:
    zip_path = os.path.join(destination_dir, file.filename)
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return zip_path

def extract_zip(zip_path: str, extract_to: str) -> None:
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def create_output_dir(pdf_path: str) -> str:
    pdf_folder_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join('./data', pdf_folder_name, 'input')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
