import dlib
import numpy as np
import cv2
import time  # Import the time module


def getFace(img):
    face_detector = dlib.get_frontal_face_detector()
    return face_detector(img, 1)[0]


def encodeFace(image):
    face_location = getFace(image)
    pose_predictor = dlib.shape_predictor(
        'models/shape_predictor_68_face_landmarks_GTX.dat')
    face_landmarks = pose_predictor(image, face_location)
    face_encoder = dlib.face_recognition_model_v1(
        'models/dlib_face_recognition_resnet_model_v1.dat')
    face = dlib.get_face_chip(image, face_landmarks)
    encodings = np.array(face_encoder.compute_face_descriptor(face))
    return encodings


def getSimilarity(image1, image2):
    face1_embeddings = encodeFace(image1)
    face2_embeddings = encodeFace(image2)
    return np.linalg.norm(face1_embeddings-face2_embeddings)


# Measure the execution time of the getSimilarity function
start_time = time.time()  # Start the timer

img1 = cv2.imread('Biden_rally_at_Bowie_State_University_(52485660899).jpeg')
img2 = cv2.imread('Joe_Biden_presidential_portrait.jpeg')

distance = getSimilarity(img1, img2)

end_time = time.time()  # End the timer
execution_time = end_time - start_time  # Calculate the execution time

print(f"Execution time: {execution_time} seconds")

if distance < .6:
    print("Faces are of the same person.")
else:
    print("Faces are of different people.")