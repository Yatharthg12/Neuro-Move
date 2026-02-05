from flask import Flask, Response
import cv2

from pose.pose_detector import get_pose_landmarks, draw_pose
from exercises.arm_raise import shoulder_angle
from exercises.sit_to_stand import knee_angle
from exercises.knee_extension import knee_extension_angle
from scoring.metrics import compute_rom, compute_smoothness, compute_consistency
from scoring.score import overall_score
from exercises.head_movement import head_horizontal_offset

app = Flask(__name__)
cap = cv2.VideoCapture(0)

# -------------------------------------------------
# CONFIG: choose exercise here
# -------------------------------------------------
ACTIVE_EXERCISE = "head_movement"
# options: "arm_raise", "sit_to_stand", "knee_extension", "head_movement"

# -------------------------------------------------
# Shared rehab state
# -------------------------------------------------
rep_count = 0
phase_active = False
ALPHA = 0.2
smoothed_angle = None

current_rep_angles = []
rep_roms = []

# Thresholds
ARM_UP = 85
ARM_DOWN = 45

STAND_UP = 160
SIT_DOWN = 100

KNEE_EXTEND = 160
KNEE_BEND = 110

HEAD_RIGHT = 40
HEAD_LEFT = -40


def gen_frames():
    global rep_count, phase_active, smoothed_angle
    global current_rep_angles, rep_roms

    while True:
        success, frame = cap.read()
        if not success:
            continue

        try:
            landmarks = get_pose_landmarks(frame)

            if landmarks is not None:
                draw_pose(frame, landmarks)

                # -----------------------------
                # Select exercise
                # -----------------------------
                if ACTIVE_EXERCISE == "arm_raise":
                    raw_angle = shoulder_angle(landmarks, frame.shape)
                    UP_TH, DOWN_TH = ARM_UP, ARM_DOWN
                    label = "Arm Raise"

                elif ACTIVE_EXERCISE == "sit_to_stand":
                    raw_angle = knee_angle(landmarks, frame.shape)
                    UP_TH, DOWN_TH = STAND_UP, SIT_DOWN
                    label = "Sit-to-Stand"

                elif ACTIVE_EXERCISE == "knee_extension":
                    raw_angle = knee_extension_angle(landmarks, frame.shape)
                    UP_TH, DOWN_TH = KNEE_EXTEND, KNEE_BEND
                    label = "Knee Extension"

                else:  # head_movement
                    raw_angle = head_horizontal_offset(landmarks, frame.shape)
                    UP_TH, DOWN_TH = HEAD_RIGHT, HEAD_LEFT
                    label = "Head Rotation"

                # -----------------------------
                # Smoothing
                # -----------------------------
                if smoothed_angle is None:
                    smoothed_angle = raw_angle
                else:
                    smoothed_angle = (
                        ALPHA * raw_angle + (1 - ALPHA) * smoothed_angle
                    )

                # Track angles during active phase
                if phase_active:
                    current_rep_angles.append(smoothed_angle)

                # -----------------------------
                # Rep detection
                # -----------------------------
                if smoothed_angle > UP_TH and not phase_active:
                    phase_active = True
                    current_rep_angles = []

                if smoothed_angle < DOWN_TH and phase_active:
                    phase_active = False
                    rep_count += 1

                    rom = compute_rom(current_rep_angles)
                    smoothness = compute_smoothness(current_rep_angles)
                    rep_roms.append(rom)
                    consistency = compute_consistency(rep_roms)
                    score = overall_score(rom, smoothness, consistency)

                    print(f"{label} — Rep {rep_count}")
                    print(f"  ROM: {int(rom)}")
                    print(f"  Smoothness: {smoothness:.2f}")
                    print(f"  Consistency: {consistency:.2f}")
                    print(f"  Score: {score}/100\n")

                # -----------------------------
                # Overlay
                # -----------------------------
                cv2.putText(
                    frame,
                    label,
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Angle: {int(smoothed_angle)} deg",
                    (30, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Reps: {rep_count}",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{label}",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 200, 0),
                    2
                )

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


@app.route("/video")
def video():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=False, threaded=True)
