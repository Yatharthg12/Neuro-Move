def overall_score(rom, smoothness, consistency):
    # Normalize roughly for demo (not medical scale)
    rom_score = min(rom / 120, 1.0)
    smooth_score = min(smoothness / 5, 1.0)
    consistency_score = min(consistency, 1.0)

    return int(
        100 * (
            0.4 * rom_score +
            0.3 * smooth_score +
            0.3 * consistency_score
        )
    )
