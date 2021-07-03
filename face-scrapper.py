#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import cv2
from mtcnn import MTCNN
import numpy as np

detector = MTCNN()

def load_args():
    parser = argparse.ArgumentParser(description="Get faces from images")
    parser.add_argument("path",
                        type=str,
                        help="Path to get faces from. If it is a directory gets all faces from all images"
                        )
    return parser.parse_args()

def get_images(path):
    if path.is_dir():
        return [cv2.imread(str(file_path)) for file_path in path.iterdir()]
    return cv2.imread(str(path))

def get_faces(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_borders = detector.detect_faces(rgb_img)
    face_borders = [face_border['box'] for face_border in face_borders]
    faces = [image[y:y+h, x:x+w] for (x, y, w, h) in face_borders]
    return faces

if __name__ == "__main__":
    path = Path(load_args().path)
    images = get_images(path)
    if isinstance(images, list):
        faces = [get_faces(image) for image in images]
        for file_name, faces in zip(path.iterdir(), faces):
            for i, face in enumerate(faces):
                cv2.imwrite(f"./faces/images/{file_name.name[:-4]}-{i}.jpg", face)
    else:
        faces = get_faces(images)
        for i, face in enumerate(faces):
            cv2.imwrite(f"./faces/images/{path.name[:-4]}-{i}.jpg", face)
