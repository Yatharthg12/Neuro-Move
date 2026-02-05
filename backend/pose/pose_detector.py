import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "movenet.tflite")
)

print("Loading model from:", MODEL_PATH)
print("Exists?", os.path.exists(MODEL_PATH))

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Force correct input shape
interpreter.resize_tensor_input(
    input_details[0]["index"],
    [1, 192, 192, 3],
    strict=True
)
interpreter.allocate_tensors()


def get_pose_landmarks(frame):
    img = cv2.resize(frame, (192, 192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])

    # MultiPose output: (1, 6, 56)
    persons = output[0]

    best_person = None
    best_score = 0

    for person in persons:
        keypoints = person[:51].reshape((17, 3))
        avg_conf = np.mean(keypoints[:, 2])
        if avg_conf > best_score:
            best_score = avg_conf
            best_person = keypoints

    return best_person


def draw_pose(frame, keypoints):
    if keypoints is None:
        return

    h, w, _ = frame.shape
    for y, x, conf in keypoints:
        if conf > 0.3:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
