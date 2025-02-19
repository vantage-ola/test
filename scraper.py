import os
import requests
from bs4 import BeautifulSoup

# Create directory if not exists
IMAGE_DIR = "./data/dataset/images/"
os.makedirs(IMAGE_DIR, exist_ok=True)

URL = "https://www.amazon.com/gp/goldbox?ref_=nav_cs_gb"

def scrape_product_images():
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(URL, headers=headers)

    if response.status_code != 200:
        print("Failed to fetch the website")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    img_tags = soup.find_all("img")

    for idx, img_tag in enumerate(img_tags[:50]):  # Limit to 50 images
        img_url = img_tag.get("src")

        if not img_url.startswith("http"):
            img_url = "https://www.amazon.com/gp/goldbox?ref_=nav_cs_gb" + img_url  # Adjust if URLs are relative

        img_data = requests.get(img_url).content
        with open(f"{IMAGE_DIR}/product_{idx}.jpg", "wb") as img_file:
            img_file.write(img_data)

        print(f"Saved: product_{idx}.jpg")

scrape_product_images()
