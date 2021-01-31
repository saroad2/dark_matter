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


def get_dv_dr(r, v, v_err):
    r_average = np.average(r)
    r2_average = np.average(r ** 2)
    v_average = np.average(v)
    rv_average = np.average(r * v)
    dv_dr_val = (rv_average - r_average * v_average) / (r2_average - r_average ** 2)
    dv_dr_err = np.sqrt(np.average(((r - r_average) * v_err) ** 2))
    return dv_dr_val, dv_dr_err


def get_partwise_dv_dr(r, v, v_err, buldge_radius):
    dv_dr_before, dv_dr_err_before = get_dv_dr(
        r=r[r < buldge_radius],
        v=v[r < buldge_radius],
        v_err=v_err[r < buldge_radius],
    )
    dv_dr_after, dv_dr_err_after = get_dv_dr(
        r=r[r >= buldge_radius],
        v=v[r >= buldge_radius],
        v_err=v_err[r >= buldge_radius],
    )
    dv_dr = np.where(r < buldge_radius, dv_dr_before, dv_dr_after)
    dv_dr_err = np.where(r < buldge_radius, dv_dr_err_before, dv_dr_err_after)
    return dv_dr, dv_dr_err


def meter_to_kiloparsec(value):
    return value * 3.24078e-20


def kiloparsec_to_meter(value):
    return value * 3.086e20


def kilogram_to_solar_mass(value):
    return value * 5.0278e-31