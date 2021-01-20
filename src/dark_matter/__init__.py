from pathlib import Path

import click
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dark_matter.constants import FIELDNAMES, R0, V0, LONGITUDE, LONGITUDE_RADIANS, VR, \
    V, R, V_ERR, R0_ERROR, R_ERR, V0_ERROR, VR_ERR
from dark_matter.util import get_v_closest, read_data_from_file, get_fwhm_of_value


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
    plt.plot(velocities, temperatures, '.b')
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
                R_ERR: r_err
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
        ecolor="r"
    )
    plt.errorbar(
        np.abs(r_list[r_list < 0]),
        np.abs(v_list[r_list < 0]),
        xerr=r_err_list[r_list < 0],
        yerr=v_err_list[r_list < 0],
        linestyle="None",
        ecolor="b"
    )
    plt.vlines([0], [-200], [200], "r")
    plt.hlines([0], [-7.5], [6.5], "b")
    plt.xlabel("R")
    plt.ylabel("V")
    plt.show()


@dark_matter.command("combine-quarters")
@click.argument("datafile", type=click.Path(exists=True, dir_okay=False))
@click.argument("outputfile", type=click.Path(dir_okay=False))
def combine_quarters(datafile, outputfile):
    df = pd.read_csv(datafile)

    v_first = f"{V}_first"
    v_err_first = f"{V_ERR}_first"
    v_forth = f"{V}_forth"
    v_err_forth = f"{V_ERR}_forth"
    v_total = f"{V}_total"
    v_stat_err_total = f"{V}_stat_err_total"
    v_measure_err_total = f"{V}_measure_err_total"
    v_err_total = f"{V}_err_total"

    first_quarter = df[df[R] > 0]
    first_quarter[v_first] = first_quarter.pop(V)
    first_quarter[v_err_first] = first_quarter.pop(V_ERR)
    print("first_quarter")
    print(first_quarter)

    forth_quarter = df[df[R] < 0]
    forth_quarter[v_forth] = np.abs(forth_quarter.pop(V))
    forth_quarter[v_err_forth] = np.abs(forth_quarter.pop(V_ERR))
    forth_quarter[R] = np.abs(forth_quarter[R])
    print("forth_quarter")
    print(forth_quarter)

    total = first_quarter[[R, R_ERR, v_first, v_err_first]].join(
        forth_quarter[[R, v_forth, v_err_forth]].set_index(R), on=R
    )
    total[v_total] = (total[v_first] + total[v_forth]) / 2
    total[v_measure_err_total] = np.sqrt(
        total[v_err_first] ** 2 + total[v_err_forth] ** 2
    )
    total[v_stat_err_total] = np.sqrt(
        (total[v_first] - total[v_total]) ** 2 + (total[v_forth] - total[v_total]) ** 2
    )
    total[v_err_total] = np.sqrt(
        total[v_measure_err_total] ** 2 + total[v_stat_err_total] ** 2
    )
    print("total")
    print(total)
    total.to_csv(outputfile, index=False)

dark_matter()
