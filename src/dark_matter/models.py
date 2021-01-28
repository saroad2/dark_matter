from eddington import fitting_function
import numpy as np

R_B = 2.37


def buldge_density(x, buldge_decay, buldge_radius=R_B):
    return np.where(x <= buldge_radius, buldge_decay / (x ** 3), 0)


def disk_density(x, disk_basic_density, disk_basic_radius):
    return disk_basic_density * np.exp(-x / disk_basic_radius)


def burket_model(x, dark_basic_density, dark_basic_radius):
    normalized_radius = x / dark_basic_radius
    return dark_basic_density / ((1 + normalized_radius) * (1 + normalized_radius ** 2))


def nfw(x, dark_basic_density, dark_basic_radius):
    normalized_radius = x / dark_basic_radius
    return dark_basic_density / (normalized_radius * (1 + normalized_radius ** 2))


@fitting_function(n=3)
def model_1(a, x):
    buldge_decay = a[0]
    disk_basic_density = a[1]
    disk_basic_radius = a[2]

    return (
        buldge_density(x, buldge_decay=buldge_decay)
        + disk_density(
            x,
            disk_basic_density=disk_basic_density,
            disk_basic_radius=disk_basic_radius,
        )
    )


@fitting_function(n=5)
def model_2(a, x):
    buldge_decay = a[0]
    disk_basic_density = a[1]
    disk_basic_radius = a[2]
    dark_basic_density = a[3]
    dark_basic_radius = a[4]

    return (
        buldge_density(x, buldge_decay=buldge_decay)
        + disk_density(
            x,
            disk_basic_density=disk_basic_density,
            disk_basic_radius=disk_basic_radius,
        )
        + burket_model(
            x,
            dark_basic_density=dark_basic_density,
            dark_basic_radius=dark_basic_radius,
        )
    )


@fitting_function(n=5)
def model_3(a, x):
    buldge_decay = a[0]
    disk_basic_density = a[1]
    disk_basic_radius = a[2]
    dark_basic_density = a[3]
    dark_basic_radius = a[4]

    return (
        buldge_density(x, buldge_decay=buldge_decay)
        + disk_density(
            x,
            disk_basic_density=disk_basic_density,
            disk_basic_radius=disk_basic_radius,
        )
        + nfw(
            x,
            dark_basic_density=dark_basic_density,
            dark_basic_radius=dark_basic_radius,
        )
    )


@fitting_function(n=6)
def model_4(a, x):
    buldge_decay = a[0]
    disk_basic_density = a[1]
    disk_basic_radius = a[2]
    sin_frequency = a[3]
    sin_amplitude_decay = a[4]
    sin_phase = a[5]

    return (
        buldge_density(x, buldge_decay=buldge_decay)
        + disk_density(
            x,
            disk_basic_density=disk_basic_density,
            disk_basic_radius=disk_basic_radius,
        )
        + np.exp(- x / sin_amplitude_decay) * np.sin(sin_frequency * x + sin_phase)
    )
