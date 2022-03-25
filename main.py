import cv2 as cv
import numpy as np
from skimage import exposure

import matplotlib.pyplot as plt


def show_img():
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    smile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_smile.xml')
    img = cv.imread('img.jpg')
    img = cv.resize(img, (300, 370))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        smiles = smile_cascade.detectMultiScale(roi_color)

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        for (ex, ey, ew, eh) in smiles:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    cv.imshow('out', img)
    cv.waitKey(0)


def show_vid():
    cap = cv.VideoCapture('D:\Downloads\Browzer\\720_vertic_339.mp4')
    while True:
        success, img = cap.read()
        # if success:
        cv.imshow('Vid', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


def show_cam():
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv.VideoCapture(0)
    cap.set(3,640)  # width
    cap.set(4,480)  # height
    cap.set(10, 150)  # brightness
    while True:
        success, img = cap.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        i = 0
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            im48 = cv.resize(gray[y:y+h, x:x+w], (48, 48))
            im48 = im48.reshape((1, 48, 48, 1))
            #cv.imshow(f'face{i}', im48)
            print(im48.shape)
            i += 1

        cv.imshow('Vid', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


def find_poses():
    img = cv.imread('pos.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    pose_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_fullbody.xml')
    poses = pose_cascade.detectMultiScale(gray)

    for (x, y, w, h) in poses:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv.imshow('Out', img)
    cv.waitKey(0)


def hist_match_cam():
    ref = cv.imread('ref.jpg')
    cap = cv.VideoCapture(0)
    while True:
        success, src = cap.read()

        multi = True if src.shape[-1] > 1 else False
        matched = exposure.match_histograms(src, ref, multichannel=multi)

        cv.imshow('Reference', ref)
        cv.imshow('Source', src)
        cv.imshow('Matched', matched)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


def match_face():
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    smile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_smile.xml')

    cap = cv.VideoCapture(0)
    while True:
        success, img = cap.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            smiles = smile_cascade.detectMultiScale(roi_color)

            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            for (ex, ey, ew, eh) in smiles:
                cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

        cv.imshow('Vid', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


def functions():
    img = cv.imread('D:\Downloads\Browzer\sl.jpg')
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('GrayScaled', imgGray)

    imgBlur = cv.GaussianBlur(img, (7, 7), 0)
    cv.imshow('Blurred', imgBlur)

    imgCanny = cv.Canny(img, 200, 200)
    cv.imshow('Canny', imgCanny)

    kernel = np.ones((5, 5), np.uint8)
    imgDilation = cv.dilate(imgCanny, kernel,iterations=1)
    cv.imshow('Dilation', imgDilation)

    imgEroded = cv.erode(imgDilation, kernel, iterations=1)
    cv.imshow('Erosion', imgEroded)
    cv.waitKey(0)


def resizing():
    img = cv.imread('D:\Downloads\Browzer\sl.jpg')

    print(img.shape)

    imgResize = cv.resize(img, (300, 300))
    cv.imshow('resized', imgResize)

    imgCropped = img[0:350, 0:200]
    cv.imshow('cropped', imgCropped)
    cv.waitKey(0)


def draw():
    img = np.zeros((512, 512, 3), np.uint8)
    cv.imshow('Image', img)
    cv.waitKey(0)


if __name__ == '__main__':
    show_cam()
