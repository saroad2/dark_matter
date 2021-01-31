from eddington import fitting_function
import numpy as np

R_B = 2.0


def buldge_density(x, buldge_initial_density, buldge_decay_power=3, buldge_radius=R_B):
    return np.where(
        x <= buldge_radius, buldge_initial_density / (x ** buldge_decay_power), 0
    )


def disk_density(
    x,
    disk_basic_density,
    disk_basic_radius,
    disk_fraction_decay=0,
    disk_exponential_decay=1,
):
    normalized_radius = x / disk_basic_radius
    return disk_basic_density * np.power(
        normalized_radius, disk_fraction_decay
    ) * np.exp(-np.power(normalized_radius, disk_exponential_decay))


def burket_model(x, dark_basic_density, dark_basic_radius):
    normalized_radius = x / dark_basic_radius
    return dark_basic_density / ((1 + normalized_radius) * (1 + normalized_radius ** 2))


def nfw(x, dark_basic_density, dark_basic_radius):
    normalized_radius = x / dark_basic_radius
    return dark_basic_density / (normalized_radius * (1 + normalized_radius ** 2))


@fitting_function(n=6)
def model_1(a, x):
    buldge_initial_density = a[0]
    buldge_decay_power = a[1]
    disk_basic_density = a[2]
    disk_basic_radius = a[3]
    disk_fraction_decay = a[4]
    disk_exponential_decay = a[5]

    return (
        buldge_density(
            x,
            buldge_initial_density=buldge_initial_density,
            buldge_decay_power=buldge_decay_power,
        )
        + disk_density(
            x,
            disk_basic_density=disk_basic_density,
            disk_basic_radius=disk_basic_radius,
            disk_fraction_decay=disk_fraction_decay,
            disk_exponential_decay=disk_exponential_decay,
        )
    )


@fitting_function(n=8)
def model_2(a, x):
    buldge_initial_density = a[0]
    buldge_decay_power = a[1]
    disk_basic_density = a[2]
    disk_basic_radius = a[3]
    disk_fraction_decay = a[4]
    disk_exponential_decay = a[5]
    dark_basic_density = a[6]
    dark_basic_radius = a[7]

    return (
        buldge_density(
            x,
            buldge_initial_density=buldge_initial_density,
            buldge_decay_power=buldge_decay_power,
        )
        + disk_density(
            x,
            disk_basic_density=disk_basic_density,
            disk_basic_radius=disk_basic_radius,
            disk_fraction_decay=disk_fraction_decay,
            disk_exponential_decay=disk_exponential_decay,
        )
        + burket_model(
            x,
            dark_basic_density=dark_basic_density,
            dark_basic_radius=dark_basic_radius,
        )
    )


@fitting_function(n=8)
def model_3(a, x):
    buldge_initial_density = a[0]
    buldge_decay_power = a[1]
    disk_basic_density = a[2]
    disk_basic_radius = a[3]
    disk_fraction_decay = a[4]
    disk_exponential_decay = a[5]
    dark_basic_density = a[6]
    dark_basic_radius = a[7]

    return (
        buldge_density(
            x,
            buldge_initial_density=buldge_initial_density,
            buldge_decay_power=buldge_decay_power,
        )
        + disk_density(
            x,
            disk_basic_density=disk_basic_density,
            disk_basic_radius=disk_basic_radius,
            disk_fraction_decay=disk_fraction_decay,
            disk_exponential_decay=disk_exponential_decay,
        )
        + nfw(
            x,
            dark_basic_density=dark_basic_density,
            dark_basic_radius=dark_basic_radius,
        )
    )


@fitting_function(n=6)
def model_4(a, x):
    buldge_initial_density = a[0]
    disk_basic_density = a[1]
    disk_basic_radius = a[2]
    sin_frequency = a[3]
    sin_amplitude_decay = a[4]
    sin_phase = a[5]

    return (
        buldge_density(x, buldge_initial_density=buldge_initial_density)
        + disk_density(
            x,
            disk_basic_density=disk_basic_density,
            disk_basic_radius=disk_basic_radius,
        )
        + np.exp(- x / sin_amplitude_decay) * np.sin(sin_frequency * x + sin_phase)
    )
