"""
   Name    : image-gather.py
   Author  : Ian Gomez
   Date    : June 26, 2021
   Description : Module for collecting images from
   Github  : imgomez0127@github
"""

import argparse
import random
import string
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


def _get_thumbnail_urls(api_key: str, name: str, max_photo_amt:int) -> list[str]:
    """ Uses google search engine to find image urls for given query

    This function utilizes a customized search engine from the google search
    api to find images of kpop stars. This gathers images until either no
    more queries are possible due to search engine limitations, or you reach
    the maximum amount of photos gathered. If there are no more queries due to
    limitations the program will return all the gathered urls.

    Args:
    api_key (str): Key to utilize the google search api
    name (str): Query of kpop star to gather images for
    max_photo_amt (int): Max number of photos to be gathered

    Returns:
    thumbnail_urls (list[str]):  List of all the urls to download images from
    """
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
    while len(thumbnail_urls) < max_photo_amt:
        print(f"Query index {params['start']}", end='\r')
        try:
            response_package = requests.get(search_url, params=params)
            response_package.raise_for_status()
            response = response_package.json()
            urls = [content["link"] for content in response["items"]]
            thumbnail_urls.extend(urls)
            # Set query params to offset the next query so we can gather new images
            params["start"] = response["queries"]["nextPage"][0]["startIndex"]
        except requests.exceptions.HTTPError as e:
            traceback.print_exc()
            print(f"Unable to query index {params['start']}")
            print("Saving currently queried images")
            break
    return thumbnail_urls


def _save_images(thumbnail_urls):
    """ Saves images from given urls

    Requests the images from the given urls,
    It then saves the images to a randomized path.

    Args:
    thumbnail_urls (list[str]): List of urls to save images from
    """
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
        except (PIL.UnidentifiedImageError, requests.exceptions.HTTPError):
            print(f"WARNING: Couldn't load image skipping url {url}")
    print(f'Saved {len(thumbnail_urls)} images')


def main():
    args = _load_args()
    name = args.name
    max_photo_amt = args.number
    api_key = _load_key("key.txt")
    thumbnail_urls = _get_thumbnail_urls(api_key, name, max_photo_amt)
    _save_images(thumbnail_urls)


if __name__ == "__main__":
    main()
