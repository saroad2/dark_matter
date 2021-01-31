from pathlib import Path

import click
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import unumpy

from dark_matter.constants import (
    FIELDNAMES,
    R0,
    V0,
    LONGITUDE,
    LONGITUDE_RADIANS,
    VR,
    V,
    R,
    V_ERR,
    R0_ERROR,
    R_ERR,
    V0_ERROR,
    VR_ERR, V_FIRST, V_ERR_FIRST, V_FORTH, V_ERR_FORTH, V_TOTAL, V_MEASURE_ERR_TOTAL,
    V_STAT_ERR_TOTAL, V_ERR_TOTAL, G, DENSITY, DENSITY_ERR, BULDGE_RADIUS,
)
from dark_matter.util import get_v_closest, read_data_from_file, get_fwhm_of_value, \
    meter_to_kiloparsec, kiloparsec_to_meter, kilogram_to_solar_mass, get_dv_dr, \
    get_partwise_dv_dr


@click.group()
def dark_matter():
    pass


@dark_matter.command("plot-long-data")
@click.argument("datafile", type=click.Path(exists=True, dir_okay=False))
def plot_long_data(datafile):
    datafile = Path(datafile)
    velocities, temperatures = read_data_from_file(datafile)
    longitude = float(datafile.stem.split("_")[0])
    vr, tr = get_v_closest(longitude, velocities, temperatures)
    vh, th = get_fwhm_of_value(vr, tr, velocities, temperatures)
    plt.plot(velocities, temperatures, ".b")
    plt.scatter([vr], [tr], s=30, color="r")
    plt.scatter([vh], [th], s=30, color="g")
    plt.show()


@dark_matter.command("build-data")
@click.argument("datadir", type=click.Path(exists=True, file_okay=False))
@click.argument("outputfile", type=click.Path(dir_okay=False))
def build_data(datadir, outputfile):
    datadir = Path(datadir)
    data = []
    for datafile in datadir.glob("*.csv"):
        velocities, temperatures = read_data_from_file(datafile)
        longitude = float(datafile.stem.split("_")[0])
        vr, tr = get_v_closest(
            longitude=longitude, velocities=velocities, temperatures=temperatures
        )
        vh, th = get_fwhm_of_value(vr, tr, velocities, temperatures)
        vr_err = 2 * np.abs(vh - vr) / 2.355
        longitude_radians = longitude * np.pi / 180
        sin_long = np.sin(longitude_radians)
        r = R0 * sin_long
        r_err = np.abs(R0_ERROR * sin_long)
        v = vr + V0 * sin_long
        v_err = np.sqrt(vr_err ** 2 + (V0_ERROR * sin_long) ** 2)
        data.append(
            {
                LONGITUDE: longitude,
                LONGITUDE_RADIANS: longitude_radians,
                VR: vr,
                VR_ERR: vr_err,
                V: v,
                V_ERR: v_err,
                R: r,
                R_ERR: r_err,
            }
        )
    data = sorted(data, key=lambda datadict: datadict[R])
    with open(outputfile, mode="w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=FIELDNAMES)
        writer.writeheader()
        for datadict in data:
            writer.writerow(datadict)


@dark_matter.command("plot-data")
@click.argument("datafile", type=click.Path(exists=True, dir_okay=False))
def plot_data(datafile):
    data = []
    with open(datafile, mode="r") as fd:
        reader = csv.DictReader(fd)
        for row in reader:
            data.append(
                [float(row[R]), float(row[R_ERR]), float(row[V]), float(row[V_ERR])]
            )
    data = np.array(data)
    data = data.transpose()
    r_list, r_err_list, v_list, v_err_list = data
    plt.errorbar(
        r_list[r_list > 0],
        v_list[r_list > 0],
        xerr=r_err_list[r_list > 0],
        yerr=v_err_list[r_list > 0],
        linestyle="None",
        ecolor="r",
    )
    plt.errorbar(
        np.abs(r_list[r_list < 0]),
        np.abs(v_list[r_list < 0]),
        xerr=r_err_list[r_list < 0],
        yerr=v_err_list[r_list < 0],
        linestyle="None",
        ecolor="b",
    )
    plt.vlines([0], [-200], [200], "k")
    plt.hlines([0], [-7.5], [6.5], "k")
    plt.xlabel("R")
    plt.ylabel("V")
    plt.show()


@dark_matter.command("combine-quarters")
@click.argument("datafile", type=click.Path(exists=True, dir_okay=False))
@click.argument("outputfile", type=click.Path(dir_okay=False))
def combine_quarters(datafile, outputfile):
    df = pd.read_csv(datafile)

    first_quarter = df[df[R] > 0]
    first_quarter[V_FIRST] = first_quarter.pop(V)
    first_quarter[V_ERR_FIRST] = first_quarter.pop(V_ERR)
    print("first_quarter")
    print(first_quarter)

    forth_quarter = df[df[R] < 0]
    forth_quarter[V_FORTH] = np.abs(forth_quarter.pop(V))
    forth_quarter[V_ERR_FORTH] = np.abs(forth_quarter.pop(V_ERR))
    forth_quarter[R] = np.abs(forth_quarter[R])
    print("forth_quarter")
    print(forth_quarter)

    total = first_quarter[[R, R_ERR, V_FIRST, V_ERR_FIRST]].join(
        forth_quarter[[R, V_FORTH, V_ERR_FORTH]].set_index(R), on=R
    )
    total[V_TOTAL] = (total[V_FIRST] + total[V_FORTH]) / 2
    total[V_MEASURE_ERR_TOTAL] = np.sqrt(
        total[V_ERR_FIRST] ** 2 + total[V_ERR_FORTH] ** 2
    ) / 2
    total[V_STAT_ERR_TOTAL] = np.sqrt(
        (total[V_FIRST] - total[V_TOTAL]) ** 2 + (total[V_FORTH] - total[V_TOTAL]) ** 2
    ) / np.sqrt(2)
    total[V_ERR_TOTAL] = np.sqrt(
        total[V_MEASURE_ERR_TOTAL] ** 2 + total[V_STAT_ERR_TOTAL] ** 2
    )
    print("total")
    print(total)
    total.to_csv(outputfile, index=False)


@dark_matter.command("monotonic-quarters")
@click.argument("datafile", type=click.Path(exists=True, dir_okay=False))
@click.argument("outputfile", type=click.Path(dir_okay=False))
def monotonic_quarters(datafile, outputfile):
    df = pd.read_csv(datafile)
    max_v = 0
    selected_indices = []
    for index, record in df.iterrows():
        v_total = float(record[V_TOTAL])
        if v_total >= max_v:
            max_v = v_total
            selected_indices.append(index)
    new_df = df.iloc[selected_indices]
    new_df.to_csv(outputfile, index=False)


@dark_matter.command("calculate-dv-dr")
@click.argument("datafile", type=click.Path(exists=True, dir_okay=False))
# @click.argument("outputfile", type=click.Path(dir_okay=False))
@click.option("-b", "--buldge-radius", type=int, default=BULDGE_RADIUS)
def calculate_dv_dr(datafile, buldge_radius):
    df = pd.read_csv(datafile)
    r, v, v_err = df[R], df[V_TOTAL], df[V_ERR_TOTAL]
    dv_dr, dv_dr_err = get_partwise_dv_dr(
        r=r, v=v, v_err=v_err, buldge_radius=buldge_radius
    )
    plt.plot(r, dv_dr)
    plt.xlabel("R")
    plt.ylabel("V")
    plt.show()


@dark_matter.command("calculate-density")
@click.argument("datafile", type=click.Path(exists=True, dir_okay=False))
@click.argument("outputfile", type=click.Path(dir_okay=False))
@click.option("-b", "--buldge-radius", type=int, default=BULDGE_RADIUS)
def calculate_density(datafile, outputfile, buldge_radius):
    df = pd.read_csv(datafile)
    r = df[R]
    r_err = df[R_ERR]
    v_total = meter_to_kiloparsec(1_000 * df[V_TOTAL])
    v_err_total = meter_to_kiloparsec(1_000 * df[V_ERR_TOTAL])
    dv_dr_val, dv_dr_err = get_partwise_dv_dr(
        r=r, v=v_total, v_err=v_err_total, buldge_radius=buldge_radius
    )

    g = meter_to_kiloparsec(G)
    r_unc = unumpy.uarray(r, r_err)
    v_unc = unumpy.uarray(v_total, v_err_total)
    dv_dr_unc = unumpy.uarray(dv_dr_val, dv_dr_err)

    density = kilogram_to_solar_mass(
        1 / (4 * g * np.pi * r_unc ** 2) * (v_unc ** 2 + 2 * r_unc * dv_dr_unc)
    )
    data = {
        R: r,
        R_ERR: r_err,
        DENSITY: unumpy.nominal_values(density),
        DENSITY_ERR: unumpy.std_devs(density),
    }

    new_df = pd.DataFrame(data, columns=[R, R_ERR, DENSITY, DENSITY_ERR])

    print("Writing...")
    with open(outputfile, mode="w", newline="") as fd:
        new_df.to_csv(fd, index=False)
    print("Done!")


@dark_matter.command("plot-quarters")
@click.argument("datafile", type=click.Path(exists=True, dir_okay=False))
@click.option("-t", "--total-only", is_flag=True)
def plot_quarters(datafile, total_only):
    df = pd.read_csv(datafile)
    if not total_only:
        plt.errorbar(
            df[R],
            df[V_FIRST],
            xerr=df[R_ERR],
            yerr=df[V_ERR_FIRST],
            linestyle="None",
            ecolor="b",
        )
        plt.errorbar(
            df[R],
            df[V_FORTH],
            xerr=df[R_ERR],
            yerr=df[V_ERR_FORTH],
            linestyle="None",
            ecolor="r",
        )
    plt.errorbar(
        df[R],
        df[V_TOTAL],
        xerr=df[R_ERR],
        yerr=df[V_ERR_TOTAL],
        linestyle="None",
        ecolor="y",
    )
    plt.vlines([0], [-200], [200], "k")
    plt.hlines([0], [-7.5], [6.5], "k")
    plt.xlabel("R")
    plt.ylabel("V")
    plt.show()


@dark_matter.command("plot-density")
@click.argument("datafile", type=click.Path(exists=True, dir_okay=False))
def plot_density(datafile):
    df = pd.read_csv(datafile)
    plt.errorbar(
        df[R],
        df[DENSITY],
        xerr=df[R_ERR],
        yerr=df[DENSITY_ERR],
        linestyle="None",
    )
    plt.yscale("log")
    plt.xlabel("R")
    plt.ylabel(r"$\rho$")
    plt.show()


dark_matter()
