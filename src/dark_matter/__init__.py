from pathlib import Path

import click
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import ufloat

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
    V_STAT_ERR_TOTAL, V_ERR_TOTAL, G, DENSITY, DENSITY_ERR,
)
from dark_matter.util import get_v_closest, read_data_from_file, get_fwhm_of_value, \
    meter_to_kiloparsec, kiloparsec_to_meter, kilogram_to_solar_mass


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
    )
    total[V_ERR_TOTAL] = np.sqrt(
        total[V_MEASURE_ERR_TOTAL] ** 2 + total[V_STAT_ERR_TOTAL] ** 2
    )
    print("total")
    print(total)
    total.to_csv(outputfile, index=False)


@dark_matter.command("calculate-density")
@click.argument("datafile", type=click.Path(exists=True, dir_okay=False))
@click.argument("outputfile", type=click.Path(dir_okay=False))
@click.option("-n", "--number-of-values", type=int, default=3)
def calculate_density(datafile, outputfile, number_of_values):
    df = pd.read_csv(datafile)
    data = []
    for index in range(number_of_values, df.shape[0] - number_of_values):
        records = df.iloc[index - number_of_values: index + number_of_values + 1]
        r = records[R]
        r_err = records[R_ERR]
        v_total = meter_to_kiloparsec(1_000 * records[V_TOTAL])
        v_err_total = meter_to_kiloparsec(1_000 * records[V_ERR_TOTAL])

        r_average = np.average(r)
        r2_average = np.average(r ** 2)
        v_average = np.average(v_total)
        rv_average = np.average(r * v_total)
        dv_dr_val = (rv_average - r_average * v_average) / (r2_average - r_average ** 2)
        dv_dr_err = np.average(((r - r_average) * v_err_total) ** 2)

        g = meter_to_kiloparsec(G)

        r_unc = ufloat(r[index], r_err[index])
        v_unc = ufloat(v_total[index], v_err_total[index])
        dv_dr_unc = ufloat(dv_dr_val, dv_dr_err)
        density = 1 / (4 * g * np.pi * r_unc ** 2) * (v_unc ** 2 + 2 * r_unc * dv_dr_unc)
        # if density < 0:
        #     print("Got negative density. Ignoring and moving on")
        #     continue
        data.append(
            {
                R: r_unc.n,
                R_ERR: r_unc.s,
                DENSITY: kilogram_to_solar_mass(density.n),
                DENSITY_ERR: kilogram_to_solar_mass(density.s)
            }
        )
    print("Writing...")
    with open(outputfile, mode="w", newline="") as fd:
        csv_writer = csv.DictWriter(fd, fieldnames=[R, R_ERR, DENSITY, DENSITY_ERR])
        csv_writer.writeheader()
        csv_writer.writerows(data)
    print("Done!")


@dark_matter.command("plot-quarters")
@click.argument("datafile", type=click.Path(exists=True, dir_okay=False))
def plot_quarters(datafile):
    df = pd.read_csv(datafile)
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


dark_matter()
