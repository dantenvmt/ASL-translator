# Import necessary libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque, Counter
import mediapipe as mp
from text_to_speech import TextToSpeech

#load model
model = tf.keras.models.load_model('asl_combined_model.h5')
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G',
    'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y', 'Z', 'del',
    'nothing', 'space'
]
#look for hand gesture
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils
#open cam
cap = cv2.VideoCapture(0)
imageSize = 64
prediction_history = deque(maxlen=15)
#add text to speech
tts = TextToSpeech()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, c = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w)
            y_min = int(min(y_coords) * h)
            x_max = int(max(x_coords) * w)
            y_max = int(max(y_coords) * h)
            margin = 20
            x1 = max(0, x_min - margin)
            y1 = max(0, y_min - margin)
            x2 = min(w, x_max + margin)
            y2 = min(h, y_max + margin)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi_resized = cv2.resize(roi, (imageSize, imageSize))
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            roi_normalized = roi_rgb / 255.0
            roi_expanded = np.expand_dims(roi_normalized, axis=0)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks)
            landmarks_expanded = landmarks.reshape(1, -1)
            predictions = model.predict([roi_expanded, landmarks_expanded])
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_label = class_labels[predicted_class_index]
            prediction_history.append(predicted_class_label)
            most_common_prediction = Counter(prediction_history).most_common(1)[0][0]
            cv2.putText(
                frame,
                f'Prediction: {most_common_prediction}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            tts.speak(most_common_prediction)
    else:
        prediction_history.clear()
        tts.reset()
    cv2.imshow('ASL Real-Time Recognition', frame)
    #press f to stop
    keypress = cv2.waitKey(1)
    if keypress & 0xFF == ord('f'):
        break
cap.release()
cv2.destroyAllWindows()
