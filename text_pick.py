
import cv2
import numpy as np 
# from PIL import Image
import os
import pytesseract


def get_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert the colors
    inverted = 255 - gray
    return inverted


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(self, image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(
        image, template, cv2.TM_CCOEFF_NORMED)


if __name__ == "__main__":
    path = os.getcwd()
    path = 'files/DmMElrzXgAYGOJz.jpg'
    if os.path.isfile(path):
        try:
            file_extension = os.path.splitext(path)[1]
            if file_extension.lower() in {'.pdf'}:
                pass
            else:
                img = cv2.imread(path)
                gray = get_grayscale(img)
                thresh = thresholding(gray)
                custom_config = r'--oem 3 --psm 6'
                scanned_contents = pytesseract.image_to_string(
                    img, config=custom_config)
                print(scanned_contents)
        except Exception as e:
            print(f'Error in file. {e}')
