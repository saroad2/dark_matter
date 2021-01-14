from pathlib import Path

import click
import csv
import numpy as np
import matplotlib.pyplot as plt

from dark_matter.constants import FIELDNAMES, R0, V0, LONGITUDE, LONGITUDE_RADIANS, VR, \
    V, R
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
        vr, _ = get_v_closest(
            longitude=longitude, velocities=velocities, temperatures=temperatures
        )
        longitude_radians = longitude * np.pi / 180
        r = R0 * np.sin(longitude_radians)
        v = vr + V0 * np.sin(longitude_radians)
        data.append(
            {
                LONGITUDE: longitude,
                LONGITUDE_RADIANS: longitude_radians,
                VR: vr,
                V: v,
                R: r
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
            data.append([float(row[R]), float(row[V])])
    data = np.array(data)
    data = data.transpose()
    r_list, v_list = data[0], data[1]
    plt.plot(r_list[r_list > 0], v_list[r_list > 0], ".k")
    plt.plot(np.abs(r_list[r_list < 0]), np.abs(v_list[r_list < 0]), ".b")
    plt.vlines([0], [-200], [200], "r")
    plt.hlines([0], [-7.5], [6.5], "b")
    plt.xlabel("R")
    plt.ylabel("V")
    plt.show()


dark_matter()
