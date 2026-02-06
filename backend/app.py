from flask import Flask, Response, request, jsonify, render_template
from flask_cors import CORS

import cv2
import time

from pose.pose_detector import get_pose_landmarks, draw_pose
from exercises.arm_raise import shoulder_angle
from exercises.sit_to_stand import knee_angle
from exercises.knee_extension import knee_extension_angle
from exercises.head_movement import head_horizontal_offset
from scoring.metrics import compute_rom, compute_smoothness, compute_consistency
from scoring.score import overall_score
from ai.features import extract_features
from ai.infer import predict_quality


app = Flask(__name__)
cap = cv2.VideoCapture(0)
CORS(app)

# -------------------------------------------------
# GLOBAL EXERCISE STATE (frontend controlled)
# -------------------------------------------------
current_exercise = "arm_raise"
baseline_shoulder_width = None

# -------------------------------------------------
# Shared rehab state
# -------------------------------------------------
rep_count = 0
phase_active = False
ALPHA = 0.2
smoothed_angle = None

current_rep_angles = []
rep_roms = []
rep_start_time = 0

# -------------------------------------------------
# Thresholds
# -------------------------------------------------
ARM_UP = 85
ARM_DOWN = 45

STAND_UP = 160
SIT_DOWN = 100

KNEE_EXTEND = 160
KNEE_BEND = 110

HEAD_RIGHT = 40
HEAD_LEFT = -40

# Depth thresholds (ADDED)
MIN_DEPTH_RATIO = 0.7
MAX_DEPTH_RATIO = 1.3

EXERCISE_ID = {
    "arm_raise": 0,
    "sit_to_stand": 1,
    "knee_extension": 2,
    "head_movement": 3
}

# -------------------------------------------------
# Status object (frontend polling)
# -------------------------------------------------
current_status = {
    "score": 0,
    "feedback": "Start exercising",
    "ai": "N/A",
    "reps": 0,
    "depth": "Calibrating",
    "lighting": "Checking"
}

# -------------------------------------------------
# API: set active exercise
# -------------------------------------------------
@app.route("/set_exercise", methods=["POST"])
def set_exercise():
    global current_exercise, rep_count, phase_active
    global smoothed_angle, current_rep_angles, rep_roms
    global baseline_shoulder_width

    data = request.get_json()
    exercise = data.get("exercise")

    if exercise not in [
        "arm_raise",
        "sit_to_stand",
        "knee_extension",
        "head_movement"
    ]:
        return jsonify({"error": "Invalid exercise"}), 400

    current_exercise = exercise
    rep_count = 0
    phase_active = False
    smoothed_angle = None
    current_rep_angles = []
    rep_roms = []
    baseline_shoulder_width = None

    current_status["score"] = 0
    current_status["feedback"] = "Start exercising"
    current_status["ai"] = "N/A"
    current_status["reps"] = 0
    current_status["depth"] = "Calibrating"
    current_status["lighting"] = "Checking"

    print(f"Switched to exercise: {exercise}")

    return jsonify({"status": "ok", "exercise": exercise})


def gen_frames():
    global rep_count, phase_active, smoothed_angle
    global current_rep_angles, rep_roms, baseline_shoulder_width

    quality = "N/A"
    depth_status = "Calibrating"
    lighting_status = "Checking"

    while True:
        success, frame = cap.read()
        if not success:
            continue

        try:
            # -----------------------------
            # Lighting check (ADDED)
            # -----------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean()

            if brightness < 60:
                lighting_status = "Too dark"
            elif brightness > 200:
                lighting_status = "Too bright"
            else:
                lighting_status = "Good lighting"

            current_status["lighting"] = lighting_status

            landmarks = get_pose_landmarks(frame)

            if landmarks is not None:
                draw_pose(frame, landmarks)

                # -----------------------------
                # Depth proxy using shoulder width (ADDED)
                # -----------------------------
                left_shoulder = landmarks[5]
                right_shoulder = landmarks[6]

                shoulder_width = abs(left_shoulder[1] - right_shoulder[1]) * frame.shape[1]

                if baseline_shoulder_width is None:
                    baseline_shoulder_width = shoulder_width

                depth_ratio = shoulder_width / baseline_shoulder_width

                if depth_ratio < MIN_DEPTH_RATIO:
                    depth_status = "Too far from camera"
                elif depth_ratio > MAX_DEPTH_RATIO:
                    depth_status = "Too close to camera"
                else:
                    depth_status = "Good distance"

                current_status["depth"] = depth_status

                # -----------------------------
                # Exercise selection (UNCHANGED)
                # -----------------------------
                if current_exercise == "arm_raise":
                    raw_angle = shoulder_angle(landmarks, frame.shape)
                    UP_TH, DOWN_TH = ARM_UP, ARM_DOWN
                    label = "Arm Raise"

                elif current_exercise == "sit_to_stand":
                    raw_angle = knee_angle(landmarks, frame.shape)
                    UP_TH, DOWN_TH = STAND_UP, SIT_DOWN
                    label = "Sit-to-Stand"

                elif current_exercise == "knee_extension":
                    raw_angle = knee_extension_angle(landmarks, frame.shape)
                    UP_TH, DOWN_TH = KNEE_EXTEND, KNEE_BEND
                    label = "Knee Extension"

                else:
                    raw_angle = head_horizontal_offset(landmarks, frame.shape)
                    UP_TH, DOWN_TH = HEAD_RIGHT, HEAD_LEFT
                    label = "Head Rotation"

                # -----------------------------
                # Smoothing (UNCHANGED)
                # -----------------------------
                if smoothed_angle is None:
                    smoothed_angle = raw_angle
                else:
                    smoothed_angle = (
                        ALPHA * raw_angle + (1 - ALPHA) * smoothed_angle
                    )

                if phase_active:
                    current_rep_angles.append(smoothed_angle)

                # -----------------------------
                # Rep detection (UNCHANGED)
                # -----------------------------
                if smoothed_angle > UP_TH and not phase_active:
                    phase_active = True
                    current_rep_angles = []
                    rep_start_time = time.time()

                if smoothed_angle < DOWN_TH and phase_active:
                    phase_active = False
                    rep_count += 1

                    rom = compute_rom(current_rep_angles)
                    smoothness = compute_smoothness(current_rep_angles)
                    rep_roms.append(rom)
                    consistency = compute_consistency(rep_roms)
                    score = overall_score(rom, smoothness, consistency)

                    rep_duration = time.time() - rep_start_time
                    features = extract_features(
                        rom,
                        smoothness,
                        consistency,
                        rep_duration,
                        EXERCISE_ID[current_exercise]
                    )
                    quality = predict_quality(features)

                    if score < 45:
                        quality = "Incorrect"

                    if score < 40:
                        feedback = "Increase range and slow down the movement."
                    elif score < 70:
                        feedback = "Good effort. Focus on smoother control."
                    else:
                        feedback = "Excellent form. Keep it up."

                    current_status["score"] = int(score)
                    current_status["feedback"] = feedback
                    current_status["ai"] = quality
                    current_status["reps"] = rep_count

                # -----------------------------
                # Overlay (ONLY ADDITIONS)
                # -----------------------------
                cv2.putText(frame, label, (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

                cv2.putText(frame, f"Angle: {int(smoothed_angle)}", (30, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.putText(frame, f"Reps: {rep_count}", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                cv2.putText(frame, f"AI: {quality}", (30, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.putText(frame, f"Depth: {depth_status}", (30, 165),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                cv2.putText(frame, f"Lighting: {lighting_status}", (30, 195),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        except Exception as e:
            print("Frame processing error:", e)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    return current_status


@app.route("/reset_reps", methods=["POST"])
def reset_reps():
    global rep_count, phase_active, smoothed_angle
    global current_rep_angles, rep_roms, baseline_shoulder_width

    rep_count = 0
    phase_active = False
    smoothed_angle = None
    current_rep_angles = []
    rep_roms = []
    baseline_shoulder_width = None

    current_status["reps"] = 0
    current_status["score"] = 0
    current_status["feedback"] = "Reps reset"
    current_status["ai"] = "N/A"
    current_status["depth"] = "Calibrating"
    current_status["lighting"] = "Checking"

    return {"status": "reset"}


if __name__ == "__main__":
    app.run(debug=False, threaded=True)
