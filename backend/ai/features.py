def extract_features(rom, smoothness, consistency, duration, exercise_id):
    """
    Returns a feature vector for one rep.
    """
    return [
        rom,
        smoothness,
        consistency,
        duration,
        exercise_id
    ]
