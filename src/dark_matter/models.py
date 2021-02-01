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


def hernquist(x, basic_density, basic_radius, alpha=1, beta=1, gamma=3):
    normalized_radius = x / basic_radius
    result = basic_density / (4 * np.pi * np.power(basic_radius, 3))
    result *= np.power(normalized_radius, -alpha)
    result *= np.power(1 + np.power(normalized_radius, beta), -gamma)
    return result


def burkert_model(x, dark_basic_density, dark_basic_radius):
    normalized_radius = x / dark_basic_radius
    return dark_basic_density / ((1 + normalized_radius) * (1 + normalized_radius ** 2))


def nfw(x, dark_basic_density, dark_basic_radius):
    normalized_radius = x / dark_basic_radius
    return dark_basic_density / (normalized_radius * (1 + normalized_radius ** 2))


@fitting_function(n=3)
def astro_no_dark(a, x):
    buldge_initial_density = a[0]
    disk_basic_density = a[1]
    disk_basic_radius = a[2]

    return (
        buldge_density(
            x,
            buldge_initial_density=buldge_initial_density,
        )
        + disk_density(
            x,
            disk_basic_density=disk_basic_density,
            disk_basic_radius=disk_basic_radius,
        )
    )


@fitting_function(n=6)
def sersic_no_dark(a, x):
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


@fitting_function(n=5)
def astro_plus_burkert(a, x):
    buldge_initial_density = a[0]
    disk_basic_density = a[1]
    disk_basic_radius = a[2]
    dark_basic_density = a[3]
    dark_basic_radius = a[4]

    return (
        buldge_density(
            x,
            buldge_initial_density=buldge_initial_density,
        )
        + disk_density(
            x,
            disk_basic_density=disk_basic_density,
            disk_basic_radius=disk_basic_radius,
        )
        + burkert_model(
            x,
            dark_basic_density=dark_basic_density,
            dark_basic_radius=dark_basic_radius,
        )
    )


@fitting_function(n=8)
def sersic_plus_burkert(a, x):
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
        + burkert_model(
            x,
            dark_basic_density=dark_basic_density,
            dark_basic_radius=dark_basic_radius,
        )
    )


@fitting_function(n=5)
def astro_plus_nfw(a, x):
    buldge_initial_density = a[0]
    disk_basic_density = a[1]
    disk_basic_radius = a[2]
    dark_basic_density = a[3]
    dark_basic_radius = a[4]

    return (
        buldge_density(
            x,
            buldge_initial_density=buldge_initial_density,
        )
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


@fitting_function(n=8)
def sersic_plus_nfw(a, x):
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


@fitting_function(n=4)
def herquist_plus_burkert(a, x):
    barionic_basic_density = a[0]
    barionic_basic_radius = a[1]
    dark_basic_density = a[2]
    dark_basic_radius = a[3]

    return (
        hernquist(
            x,
            basic_density=barionic_basic_density,
            basic_radius=barionic_basic_radius,
        )
        + burkert_model(
            x,
            dark_basic_density=dark_basic_density,
            dark_basic_radius=dark_basic_radius,
        )
    )


@fitting_function(n=4)
def herquist_plus_nfw(a, x):
    barionic_basic_density = a[0]
    barionic_basic_radius = a[1]
    dark_basic_density = a[2]
    dark_basic_radius = a[3]

    return (
        hernquist(
            x,
            basic_density=barionic_basic_density,
            basic_radius=barionic_basic_radius,
        )
        + nfw(
            x,
            dark_basic_density=dark_basic_density,
            dark_basic_radius=dark_basic_radius,
        )
    )
