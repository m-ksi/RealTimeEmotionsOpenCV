import cv2 as cv
import numpy as np
import tensorflow as tf

scaler = np.vectorize(lambda x: x / 255.0)
font = cv.FONT_HERSHEY_SIMPLEX
model = tf.keras.models.load_model('model')


def preprocess(gray):
    im48 = cv.resize(gray, (48, 48))
    im48 = im48.reshape((1, 48, 48, 1))
    im48 = scaler(im48)
    return im48


def predict(im48):
    probs = model.predict(im48)
    print(probs)
    #probs = [0]*7
    return probs[0]


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
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            im48 = preprocess(gray[y:y + h, x:x + w])
            probs = predict(im48)
            dy = h//7
            for p, (prob, label) in enumerate(zip(probs, ['Angry', 'Disgust', 'Fear',
                                                          'Happy', 'Sad', 'Surprise', 'Neutral'])):
                print(p, prob, label)
                cv.putText(img, f'{label}:' + "{:.2f}".format(prob), (x+10+w, y+20+p*dy), font, 0.7, (0, 255, 0), 1)

            i += 1

        cv.imshow('Vid', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    show_cam()
