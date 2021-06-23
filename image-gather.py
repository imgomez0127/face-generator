#!/usr/bin/env python3
import random
import string
import argparse
import traceback
import requests
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from io import BytesIO

def _load_key(file_name):
    with open(file_name, "r") as f:
        key = f.readline().strip()
    return key

def _load_args():
    parser = argparse.ArgumentParser(description="Download images of a kpop stars")
    parser.add_argument("number", type=int, help="Amount of images to download")
    parser.add_argument("name", type=str, help="Kpop celeb to search images for")
    return parser.parse_args()

if __name__ == "__main__":
    args = _load_args()
    name = args.name
    number = args.number
    api_key = _load_key("key.txt")
    engine_id = "bdad4e6da3462d720"
    search_url = f"https://www.googleapis.com/customsearch/v1/siterestrict?key={api_key}&cx={engine_id}"
    params = {
        "cx": engine_id,
        "imgType": "photo",
        "key": api_key,
        "searchType": "image",
        "q": name,
        "start":1
    }
    thumbnail_urls = []
    while len(thumbnail_urls) < number:
        print(f"Query index {params['start']}", end='\r')
        try:
            response_package = requests.get(search_url, params=params)
            response_package.raise_for_status()
            response = response_package.json()
        except requests.exceptions.HTTPError as e:
            traceback.print_exc()
            print(f"Unable to query index {params['start']}")
            print("Saving currently queried images")
            break
        urls = [content["link"] for content in response["items"]]
        thumbnail_urls.extend(urls)
        params["start"] = response["queries"]["nextPage"][0]["startIndex"]
    file_length = 30
    letters = string.ascii_letters
    print(thumbnail_urls)
    for url in thumbnail_urls:
        try:
            image_bytes = requests.get(url)
            image_buf = BytesIO(image_bytes.content)
            image = Image.open(image_buf)
            random_name = ''.join((random.choice(letters) for i in range(file_length)))
            image.save(f'./training_images/{random_name}.png')
        except PIL.UnidentifiedImageError:
            print(f"WARNING: Skipping url {url}")
    print(f'Saved {len(thumbnail_urls)} images')
