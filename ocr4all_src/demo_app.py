import requests

import streamlit as st

def upload_file(file):
    ### API Endpoints
    # - **POST /upload-zip/**: Upload a ZIP file containing PDFs.
    #     - **Request**: `multipart/form-data` with the file field named `file`.
    #     - **Response**: JSON indicating success or failure.
    url = "http://localhost:8000/upload-zip/"
    files = {'file': file}
    response = requests.post(url, files=files)
    return response

if __name__ == '__main__':
    st.title('Onboard documents')
    st.write('Upload your documents here')
    uploaded_file = st.file_uploader("Choose .zip file", type="zip")
    
    # if uploaded then send it to OCR4all
    upload_file(uploaded_file)

