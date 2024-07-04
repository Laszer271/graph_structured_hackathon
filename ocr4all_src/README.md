## OCR4all Integration with FastAPI

This project provides a FastAPI service to handle ZIP files containing PDFs, convert them to grayscale images, and prepare them for processing with OCR4all.

### Folder Structure
```
ocr4all_project/
├── data/
│   ├── [pdf_file_1]/
│   │   ├── input/
│   │   │   ├── [extracted_images].png
│   ├── [pdf_file_2]/
│   │   ├── input/
│   │   │   ├── [extracted_images].png
├── logs/
│   ├── app.log
├── main.py
├── services/
│   ├── __init__.py
│   ├── file_handler.py
│   ├── pdf_processor.py
├── Dockerfile
├── requirements.txt
└── docker-compose.yml
```

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/ocr4all-fastapi.git
    cd ocr4all-fastapi
    ```
    
### Running the Application

1. **Start the Docker containers:**
    ```sh
    docker-compose up
    ```

2. **Access the FastAPI documentation:**
    Open your browser and navigate to [http://localhost:8000/docs](http://localhost:8000/docs).

### API Endpoints

- **POST /upload-zip/**: Upload a ZIP file containing PDFs.
    - **Request**: `multipart/form-data` with the file field named `file`.
    - **Response**: JSON indicating success or failure.

### Code Structure

- **main.py**: Main FastAPI application.
- **services/file_handler.py**: Functions for handling file uploads and extractions.
- **services/pdf_processor.py**: Functions for processing PDFs and converting pages to grayscale images.

### Logging

Logs are stored in `logs/app.log`.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.