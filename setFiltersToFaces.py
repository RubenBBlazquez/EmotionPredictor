import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import random
from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp

model = load_model('models/facialPointsPredictor.h5')
model_emotions = load_model('models/emotionPredictor.h5')
emotion_dict = {0: 'Ira', 1: 'Asco', 2: 'Tristeza', 3: 'Felicidad', 4: 'Sorpresa'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing_styles = mp.solutions.drawing_styles
last_emotion_predicted = ''


def get_points(image):
    image = tf.reshape(image, (96, 96, 1))
    points = model.predict(np.array([image]))[0]

    return points[0::2], points[1::2]


def enumerate_points(points: list[tuple]):
    """Enumerate points in a list of tuples each tuple item is one more index"""
    return list(enumerate(points))


def draw_rectangles(frame):
    cv2.rectangle(frame, (0, 0), (100, 100), (255, 0, 0), -1)
    cv2.putText(frame, 'Change FT', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.rectangle(frame, (frame.shape[1] - 100, 0), (frame.shape[1], 100), (255, 0, 0), -1)
    cv2.putText(frame, 'Emotion', (frame.shape[1] - 80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def get_filter(filter_image, points, color_face_redim, ancho_gafas, alto_gafas, original_shape):
    if ancho_gafas > 0 and alto_gafas > 0:
        print(ancho_gafas, alto_gafas)
        gafas_resized = cv2.resize(filter_image, (ancho_gafas, alto_gafas), interpolation=cv2.INTER_CUBIC)

        region_no_transparente = gafas_resized[:, :, :3] != 0

        color_face_redim[int(points[9][1]):int(points[9][1]) + alto_gafas,
        int(points[9][0]): int(points[9][0]) + ancho_gafas, :][region_no_transparente] = gafas_resized[:, :, :3][
            region_no_transparente]

        color_face = cv2.resize(color_face_redim, original_shape, interpolation=cv2.INTER_CUBIC)

        return color_face


def get_filter_dimensions(frame_cara, last_filter_dimensions):
    points_x, points_y = get_points(frame_cara)
    points = list(zip(points_x, points_y))

    ancho_gafas = int((points[7][0] - points[9][0]))
    alto_gafas = int((points[10][1] - points[8][1]))

    if ancho_gafas > 0 and alto_gafas > 0:
        return points, (ancho_gafas, alto_gafas)

    return False, False


def hand_detection(frame):
    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        return results.multi_hand_landmarks[0].landmark


def must_change_filter(hand_landmarks):
    if hand_landmarks is not None:
        if hand_landmarks[8].y * frame.shape[0] <= 110.0 and hand_landmarks[8].x * frame.shape[1] <= 100.0:
            return True

    return False


def must_predict_emotion(hand_landmarks):
    if hand_landmarks is not None:
        if hand_landmarks[8].y * frame.shape[0] <= 110.0 and hand_landmarks[8].x * frame.shape[1] >= frame.shape[
            1] - 100:
            return True

    return False


def predict_emotion_and_draw_emotion(frame_cara):
    frame_cara = tf.reshape(frame_cara, (-1, 105, 105, 1))
    print(frame_cara.shape)

    return emotion_dict[int(np.argmax(model_emotions.predict(frame_cara)))]


def detect_faces(frame, random_image: str, last_filter_dimensions, last_emotion_predicted):
    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    filter_image = cv2.imread(random_image, cv2.IMREAD_UNCHANGED)

    print(last_emotion_predicted)

    for (x, y, w, h) in faces:
        gray_face = gray[y:y + h, x:x + w]
        color_face = frame[y:y + h, x:x + w]
        frame_cara = cv2.resize(gray_face, (96, 96)) / 255
        frame_cara_emotion = cv2.resize(gray_face, (105, 105)) / 255
        color_face_redim = cv2.resize(color_face, (96, 96))
        original_shape = gray_face.shape

        points, last_filter_dimensions = get_filter_dimensions(frame_cara, last_filter_dimensions)

        if not last_filter_dimensions:
            continue

        ancho_gafas = last_filter_dimensions[0]
        alto_gafas = last_filter_dimensions[1]

        selfie_filter = get_filter(filter_image, points, color_face_redim, ancho_gafas, alto_gafas, original_shape)

        if selfie_filter is not None:
            frame[y:y + h, x:x + w] = selfie_filter

        if must_predict_emotion(hand_landmarks):
            print(frame_cara_emotion.shape)
            last_emotion_predicted = predict_emotion_and_draw_emotion(frame_cara_emotion)

            cv2.putText(frame, last_emotion_predicted, (frame.shape[1] // 2, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 5)



cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
handsCascade = cv2.CascadeClassifier('haarCascadeHands/cascadeHands.xml')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

last_filter_dimensions = (0, 0)
cont = 0
last_filter_used = 'filtros/sunglasses.png'
selfie_filters = ['filtros/sunglasses.png', 'filtros/sunglasses_2.png', 'filtros/sunglasses_3.jpg',
                  'filtros/sunglasses_4.png', 'filtros/sunglasses_5.jpg', 'filtros/sunglasses_6.png']

while True:
    # Capture frame-by-frame
    cap.set(cv2.CAP_PROP_FPS, 16)
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here

    frame = cv2.flip(frame, 90)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    draw_rectangles(frame)

    hand_landmarks = hand_detection(frame)

    if must_change_filter(hand_landmarks):
        last_filter_used = random.choice(selfie_filters)

    detect_faces(frame, last_filter_used, last_filter_dimensions, last_emotion_predicted)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    cont += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
