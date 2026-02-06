import numpy as np

def compute_rom(rep_angles):
    return max(rep_angles)

def compute_smoothness(rep_angles):
    velocities = np.diff(rep_angles)
    if len(velocities) == 0:
        return 0
    return 1 / (np.var(velocities) + 1e-6)

def compute_consistency(all_rep_roms):
    if len(all_rep_roms) < 2:
        return 1.0
    return 1 / (np.std(all_rep_roms) + 1e-6)
