#!/usr/bin/env python3
import random
import string
import argparse
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def load_key(file_name):
    with open(file_name, "r") as f:
        key = f.readline().strip()
    return key

def load_args():
    parser = argparse.ArgumentParser(description="Download images of a kpop stars")
    parser.add_argument("number", type=int, help="Amount of images to download")
    parser.add_argument("name", type=str, help="Kpop celeb to search images for")
    return parser.parse_args()

if __name__ == "__main__":
    args = load_args()
    name = args.name
    number = args.number
    search_url = "https://api.bing.microsoft.com/v7.0/images/search"
    subscription_key = load_key("key.txt")
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": name, "license": "public", "imageType": "photo"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    thumbnail_urls = [img['thumbnail']["thumbnailUrl"] for img in search_results["queryExpansions"][:number]]
    file_length = 10
    letters = string.ascii_letters
    for url in thumbnail_urls:
            image_data = requests.get(url)
            image_data.raise_for_status()
            image = Image.open(BytesIO(image_data.content))
            random_name = (''.join(random.choice(letters) for i in range(file_length)))
            image.save(f'./training_images/{random_name}.jpg', "JPEG")
    print(f'Saved {len(thumbnail_urls)} images')
