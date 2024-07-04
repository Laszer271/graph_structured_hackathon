import os
from typing import List
from pdf2image import convert_from_path # type: ignore

def process_pdfs(root_dir: str) -> List[str]:
    pdf_paths = []
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".pdf"):
                pdf_paths.append(os.path.join(root, filename))
    return pdf_paths

def save_images_as_gray(pdf_path: str, output_dir: str) -> None:
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        gray_image = image.convert("L")
        gray_image.save(os.path.join(output_dir, f"{i:04d}.png"))
