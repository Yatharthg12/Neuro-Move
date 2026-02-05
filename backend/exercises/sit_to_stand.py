from pose.angles import calculate_angle

def knee_angle(keypoints, frame_shape):
    """
    Computes knee angle for sit-to-stand using left leg
    Joints:
    11 -> hip
    13 -> knee
    15 -> ankle
    """
    h, w, _ = frame_shape

    hip = keypoints[11]
    knee = keypoints[13]
    ankle = keypoints[15]

    def to_xy(kp):
        return (kp[1] * w, kp[0] * h)

    return calculate_angle(
        to_xy(hip),
        to_xy(knee),
        to_xy(ankle)
    )
