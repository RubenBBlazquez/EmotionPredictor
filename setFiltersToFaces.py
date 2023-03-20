import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp


model = load_model('models/facialPointsPredictor.h5')


def get_points(image):
    image = tf.reshape(image, (96, 96, 1))
    points = model.predict(np.array([image]))[0]

    return points[0::2], points[1::2]


def enumerate_points(points: list[tuple]):
    """Enumerate points in a list of tuples each tuple item is one more index"""
    return list(enumerate(points))


def draw_rectangles(frame):
    cv2.rectangle(frame, (0, 0), (100, 100), (255, 0, 0), -1)
    cv2.putText(frame, 'Emotion', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.rectangle(frame, (frame.shape[1] - 100, 0), (frame.shape[1], 100), (255, 0, 0), -1)
    cv2.putText(frame, 'Change FT', (frame.shape[1] - 80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def get_filter(color_face_redim, ancho_gafas, alto_gafas):
    if ancho_gafas > 0 and alto_gafas > 0:
        gafas_resized = cv2.resize(gafas, (ancho_gafas, alto_gafas), interpolation=cv2.INTER_CUBIC)

        region_no_transparente = gafas_resized[:, :, :3] != 0

        x_glasess = np.arange(int(points[9][1]), int(points[9][1]) + alto_gafas)
        y_glasses = np.arange(int(points[9][0]), int(points[9][0]) + ancho_gafas)

        color_face_redim[int(points[9][1]):int(points[9][1]) + alto_gafas,
        int(points[9][0]): int(points[9][0]) + ancho_gafas, :][region_no_transparente] = gafas_resized[:, :, :3][
            region_no_transparente]

        color_face = cv2.resize(color_face_redim, original_shape, interpolation=cv2.INTER_CUBIC)

        return color_face


def hand_detection(frame):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    results = hands.process(frame)

    if results.multi_hand_landmarks:
      print(results.multi_hand_landmarks)

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
handsCascade = cv2.CascadeClassifier('haarCascadeHands/cascadeHands.xml')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cont = 0
while True:
    # Capture frame-by-frame
    cap.set(cv2.CAP_PROP_FPS, 20)
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here

    frame = cv2.flip(frame, 90)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    draw_rectangles(frame)

    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    hands = handsCascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    gafas = cv2.imread('filtros/sunglasses.png', cv2.IMREAD_UNCHANGED)

    # hand_detection(frame)
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        gray_face = gray[y:y + h, x:x + w]
        color_face = frame[y:y + h, x:x + w]
        frame_cara = cv2.resize(gray_face, (96, 96)) / 255
        color_face_redim = cv2.resize(color_face, (96, 96))
        original_shape = gray_face.shape

        points_x, points_y = get_points(frame_cara)

        points = list(zip(points_x, points_y))

        ancho_gafas = int((points[7][0] - points[9][0]))
        alto_gafas = int((points[10][1] - points[8][1]))

        selfie_filter = get_filter(color_face_redim, ancho_gafas, alto_gafas)

        if selfie_filter is not None:
            frame[y:y + h, x:x + w] = selfie_filter

        for point in enumerate_points(points):
            #resize points coordinates to original dataframe
            point_number = point[0]
            point = (point[1][0] * w / 96, point[1][1] * h / 96)

            #draw cirlce
            # cv2.circle(frame, (int(point[0] + x), int(point[1] + y)), 1, (255, 255, 255), 3)
            # set index
            #cv2.putText(frame, str(point_number), (int(point[0] + x) + 10 , int(point[1] + y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    cont += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()





