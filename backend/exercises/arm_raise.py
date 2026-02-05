from pose.angles import calculate_angle

def shoulder_angle(keypoints, frame_shape):
    h, w, _ = frame_shape

    shoulder = keypoints[5]
    elbow = keypoints[7]
    hip = keypoints[11]

    def to_xy(kp):
        return (kp[1] * w, kp[0] * h)

    return calculate_angle(
        to_xy(hip),
        to_xy(shoulder),
        to_xy(elbow)
    )
