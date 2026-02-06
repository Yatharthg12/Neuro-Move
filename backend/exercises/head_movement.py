def head_horizontal_offset(keypoints, frame_shape):
    """
    Measures left-right head movement using nose relative to shoulders.
    Returns a signed horizontal displacement value.
    """

    h, w, _ = frame_shape

    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    nose_x = nose[1] * w
    shoulder_mid_x = ((left_shoulder[1] + right_shoulder[1]) / 2) * w

    # inside head_horizontal_offset
    if left_shoulder[2] < 0.4 or right_shoulder[2] < 0.4:
        return 0.0

    return nose_x - shoulder_mid_x
