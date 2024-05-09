import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

cascPath = "/home/user/Documents/github/face_id_frappe/files/haarcascade_frontalface_default.xml"
if not os.path.isfile(cascPath):
    print(f"Error: Haar cascade file not found at {cascPath}")
else:
    faceCascade = cv2.CascadeClassifier(cascPath)
    print(f"Loaded Haar cascade from {cascPath}")
path = "files/image-1.png"
image = cv2.imread(path)
image_crop = Image.open(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=2,
    minSize=(40, 60),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

im_crop = image_crop.crop((x, y, (x+w), (y+h)))
plt.imshow(im_crop)
plt.savefig('face_detection.png')