from pose.angles import calculate_angle

def shoulder_angle(keypoints, frame_shape):
    """
    Computes shoulder flexion angle.
    Automatically selects left or right arm based on landmark confidence.
    """

    h, w, _ = frame_shape

    # MoveNet indices
    # Left: shoulder 5, elbow 7, wrist 9
    # Right: shoulder 6, elbow 8, wrist 10

    left_conf = (
        keypoints[5][2] +
        keypoints[7][2] +
        keypoints[9][2]
    )

    right_conf = (
        keypoints[6][2] +
        keypoints[8][2] +
        keypoints[10][2]
    )

    if right_conf >= left_conf:
        shoulder = keypoints[6]
        elbow = keypoints[8]
        wrist = keypoints[10]
    else:
        shoulder = keypoints[5]
        elbow = keypoints[7]
        wrist = keypoints[9]

    def to_xy(kp):
        return (kp[1] * w, kp[0] * h)

    return calculate_angle(
        to_xy(shoulder),
        to_xy(elbow),
        to_xy(wrist)
    )
