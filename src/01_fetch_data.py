import os
import urllib.request as request
from zipfile import ZipFile

data_url = "https://github.com/PriyanshuDey23/Fianacial-Stock-Analysis-with-LLama-index/raw/refs/heads/main/articles.zip"

def download_and_extract_file():
    # Download the file
    filename, headers = request.urlretrieve(url=data_url, filename="articles.zip")

    # Extract the file
    with ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall()

    # Delete the downloaded zip file
    #os.remove(filename)

download_and_extract_file()
