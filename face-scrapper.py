#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import cv2
from mtcnn import MTCNN
import numpy as np

detector = MTCNN()


def load_args():
    parser = argparse.ArgumentParser(description='Get faces from images')
    parser.add_argument(
        'path',
        type=str,
        help='Path to get faces from. If it is a directory gets all faces from all images')
    parser.add_argument(
        '--range',
        type=str,
        help='Range to extract images from format: "start stop"')
    return parser.parse_args()


def get_images(path, img_range=None):
    if path.is_dir():
        files = list(path.iterdir())
        if img_range is not None:
            files = files[img_range[0]:img_range[1]]
        for file_path in files:
            yield cv2.imread(str(file_path))
    else:
        yield cv2.imread(str(path))


def get_faces(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_borders = detector.detect_faces(rgb_img)
    face_borders = [face_border['box'] for face_border in face_borders if face_border['confidence'] > 0.9]
    faces = [image[y:y + h, x:x + w] for (x, y, w, h) in face_borders]
    return faces


if __name__ == '__main__':
    args = load_args()
    path = Path(args.path)
    range_nums = None
    file_names = path.iterdir()
    if args.range:
        range_nums = list(map(int, args.range.split(' ')))
        file_names = list(path.iterdir())[range_nums[0]:range_nums[1]]
    images = get_images(path, img_range=range_nums)
    if path.is_dir():
        faces = map(get_faces, images)
        for img_num, (file_name, faces) in enumerate(zip(path.iterdir(), faces)):
            for i, face in enumerate(faces):
                cv2.imwrite(f'./faces/new_images/{file_name.name[:-4]}-{i}.jpg',
                            face)
            print(f'Finished image {img_num}')
    else:
        faces = get_faces(images)
        for i, face in enumerate(faces):
            cv2.imwrite(f'./faces/new_images/{path.name[:-4]}-{i}.jpg', face)
