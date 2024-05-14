import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import string
import face_recognition


# Global variable def
cascPath = "files/haarcascade_frontalface_default.xml"
path = "files/image-1.png"


def crop_image():
    """
    This fucntion will use opencv to:
    1) identify subject face
    2) crop subject face
    3) store subject face
    """
    try:

        if not os.path.isfile(cascPath):
            raise ValueError(f"Error: Haar cascade file not found at {cascPath}")
        else:
            faceCascade = cv2.CascadeClassifier(cascPath)
            print(f"Loaded Haar cascade from {cascPath}")
        image = cv2.imread(path)
        image_crop = Image.open(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(40, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        print("Found {0} faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        im_crop = image_crop.crop((x, y, (x+w), (y+h)))
        plt.imshow(im_crop)
        random_name = ''.join(random.choices(
            string.ascii_letters + string.digits, k=8
            ))
        plt.savefig(f'face_crop_{random_name}.png')
        return (f'face_crop_{random_name}.png')
    except Exception as e:
        print(f"An error occurred: {e}")


def compare_face():
    """
    this function uses a package
    wth dlib at its core to compare faces
    """
    known_image = face_recognition.load_image_file("biden.jpg")
    unknown_image = face_recognition.load_image_file("unknown.jpg")

    known_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces(
        [known_encoding], unknown_encoding
        )
    print(results)