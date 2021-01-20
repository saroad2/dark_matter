import numpy as np
from scipy.signal import argrelextrema

from dark_matter.constants import THRESHOLD


def read_data_from_file(datafile):
    data = np.genfromtxt(datafile, delimiter=",")
    data = data.transpose()

    velocities = data[0]
    temperatures = data[1]
    return velocities, temperatures


def get_v_closest(longitude, velocities, temperatures):
    local_maxima_indices = argrelextrema(temperatures, np.greater)[0].tolist()
    velocities = velocities[local_maxima_indices]
    temperatures = temperatures[local_maxima_indices]
    candidates = [(v, t) for v, t in zip(velocities, temperatures) if t >= THRESHOLD]
    candidates = sorted(candidates, key=lambda v_t_tuple: v_t_tuple[0])
    if longitude > 0:
        return candidates[-1]
    else:
        return candidates[0]


def get_fwhm_of_value(vr, tr, velocities, temperatures, eps=5):
    candidates = [
        (v, t) for v, t in zip(velocities, temperatures) if np.abs(t - (tr / 2)) <= eps
    ]
    return min(candidates, key=lambda x: abs(x[0] - vr))
