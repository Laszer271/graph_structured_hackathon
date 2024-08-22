import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm

# URL of the website
url = "https://www.jabholco.com/section/press-releases"

# Send a GET request to the website
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find all rows in the table that contain the PDFs
rows = soup.find("div", class_="docs-page").find('table').find("tbody").find_all('tr')
print(f"Found {len(rows)} PDFs to download.")

# Create a directory to save the downloaded PDFs
os.makedirs('pdf_downloads', exist_ok=True)

# Loop through each row and download the PDF
pbar = tqdm(rows)
for row in pbar:
    pdf_link = row.find('a')['href']
    pdf_name = row.find('a').text.strip() + ".pdf"
    pbar.set_description(f"Downloading: {pdf_name[:20]}")

    # Complete URL of the PDF file
    pdf_url = f"https://www.jabholco.com{pdf_link}"
    
    # Send a GET request to download the PDF
    pdf_response = requests.get(pdf_url)
    
    # Save the PDF to the directory
    with open(os.path.join('pdf_downloads', pdf_name), 'wb') as f:
        f.write(pdf_response.content)
    
    # print(f"Downloaded: {pdf_name}")

print("All PDFs have been downloaded.")