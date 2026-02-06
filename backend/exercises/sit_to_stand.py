from pose.angles import calculate_angle

def knee_angle(keypoints, frame_shape):
    """
    Sit-to-stand knee angle.
    Auto-selects left or right leg.
    """

    h, w, _ = frame_shape

    # Left: hip 11, knee 13, ankle 15
    # Right: hip 12, knee 14, ankle 16

    left_conf = keypoints[11][2] + keypoints[13][2] + keypoints[15][2]
    right_conf = keypoints[12][2] + keypoints[14][2] + keypoints[16][2]

    if right_conf >= left_conf:
        hip, knee, ankle = keypoints[12], keypoints[14], keypoints[16]
    else:
        hip, knee, ankle = keypoints[11], keypoints[13], keypoints[15]

    def to_xy(kp):
        return (kp[1] * w, kp[0] * h)

    return calculate_angle(
        to_xy(hip),
        to_xy(knee),
        to_xy(ankle)
    )
