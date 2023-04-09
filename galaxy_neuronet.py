import argparse
import math
import os
import random
import urllib.request
import urllib.error
import pathlib
import numpy
import cv2
from sklearn import model_selection
from keras import models, layers


def select_galaxies():
    ell_count = 0
    sp_count = 0
    edge_count = 0
    with open("../GalaxyZoo1_DR_table2.csv", "r") as inp:
        with open("../elliptical.txt", "w") as ell:
            with open("../spiral.txt", "w") as sp:
                with open("../edge.txt", "w") as edge:
                    for s in inp:
                        vals = s.split(",")
                        ra = vals[1].split(":")
                        dec = vals[2].split(":")
                        if vals[4] != "P_EL" and float(vals[4]) > 0.9:
                            ell.write(f"nearest:B {ra[0]} {ra[1]} {ra[2]} {dec[0]} {dec[1]} {dec[2]}\n")
                            ell_count += 1
                        if vals[5] != "P_CW" and (float(vals[5]) > 0.75 or float(vals[6]) > 0.75):
                            sp.write(f"nearest:B {ra[0]} {ra[1]} {ra[2]} {dec[0]} {dec[1]} {dec[2]}\n")
                            sp_count += 1
                        if vals[7] != "P_EDGE" and float(vals[7]) > 0.8:
                            edge.write(f"nearest:B {ra[0]} {ra[1]} {ra[2]} {dec[0]} {dec[1]} {dec[2]}\n")
                            edge_count += 1
    print(f"Found {ell_count} elliptical,{sp_count} spiral and {edge_count} edge galaxies")


def filter_galaxies(d_min):
    with open("../elliptical_data.txt", "r") as inp:
        with open("elliptical_set.txt", "w") as out:
            for s in inp:
                if s.startswith("nearest"):
                    vals = s.split("|")
                    if not vals[4].isspace():
                        d = math.pow(10, float(vals[4])) * 6
                        if d_min < d:
                            coord = vals[0].split()
                            out.write(f"{coord[1]} {coord[2]} {coord[3]} {coord[4]} {coord[5]} {coord[6]} {vals[1]} {d}\n")
    with open("../spiral_data.txt", "r") as inp:
        with open("spiral_set.txt", "w") as out:
            for s in inp:
                if s.startswith("nearest"):
                    vals = s.split("|")
                    if not vals[4].isspace():
                        d = math.pow(10, float(vals[4])) * 6
                        if d > d_min:
                            coord = vals[0].split()
                            out.write(f"{coord[1]} {coord[2]} {coord[3]} {coord[4]} {coord[5]} {coord[6]} {vals[1]} {d}\n")
    with open("../edge_data.txt", "r") as inp:
        with open("edge_set.txt", "w") as out:
            for s in inp:
                if s.startswith("nearest"):
                    vals = s.split("|")
                    if not vals[4].isspace():
                        d = math.pow(10, float(vals[4])) * 6
                        if d > d_min:
                            coord = vals[0].split()
                            out.write(f"{coord[1]} {coord[2]} {coord[3]} {coord[4]} {coord[5]} {coord[6]} {vals[1]} {d}\n")


def download_images():
    if not os.path.exists("../spiral"):
        os.makedirs("../spiral")
    if not os.path.exists("../elliptical"):
        os.makedirs("../elliptical")
    if not os.path.exists("../edge"):
        os.makedirs("../edge")
    scale = 0.396127
    with open("elliptical_set.txt", "r") as inp:
        i = 0
        for s in inp:
            if not os.path.exists(f"../elliptical/{i}.jpg"):
                vals = s.split()
                ra = f"{vals[0]}:{vals[1]}:{vals[2]}"
                dec = f"{vals[3]}:{vals[4]}:{vals[5]}"
                d = float(vals[7])
                size = math.ceil(1.5 * d / scale)
                url = f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale={scale}&width={size}&height={size}"
                print(url)
                try:
                    urllib.request.urlretrieve(url, f"../elliptical/{i}.jpg")
                except urllib.error.HTTPError as ex:
                    print("SDSS SkyServer cannot answer your request!")
                    print(ex)
            i += 1
    with open("spiral_set.txt", "r") as inp:
        i = 0
        for s in inp:
            if not os.path.exists(f"../spiral/{i}.jpg"):
                vals = s.split()
                ra = f"{vals[0]}:{vals[1]}:{vals[2]}"
                dec = f"{vals[3]}:{vals[4]}:{vals[5]}"
                d = float(vals[7])
                size = math.ceil(1.5 * d / scale)
                url = f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale={scale}&width={size}&height={size}"
                print(url)
                try:
                    urllib.request.urlretrieve(url, f"../spiral/{i}.jpg")
                except urllib.error.HTTPError as ex:
                    print("SDSS SkyServer cannot answer your request!")
                    print(ex)
            i += 1
    with open("edge_set.txt", "r") as inp:
        i = 0
        for s in inp:
            if not os.path.exists(f"../edge/{i}.jpg"):
                vals = s.split()
                ra = f"{vals[0]}:{vals[1]}:{vals[2]}"
                dec = f"{vals[3]}:{vals[4]}:{vals[5]}"
                d = float(vals[7])
                size = math.ceil(1.5 * d / scale)
                url = f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale={scale}&width={size}&height={size}"
                print(url)
                try:
                    urllib.request.urlretrieve(url, f"../edge/{i}.jpg")
                except urllib.error.HTTPError as ex:
                    print("SDSS SkyServer cannot answer your request!")
                    print(ex)
            i += 1


def make_model():
    model = models.Sequential()
    model.add(layers.Conv2D(4, kernel_size=3, activation="relu", input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(4, kernel_size=3, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Conv2D(8, kernel_size=3, activation="relu", input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(8, kernel_size=3, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Conv2D(4, kernel_size=3, activation="relu", input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(4, kernel_size=3, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None))
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def make_dataset():
    elliptical_galaxies = []
    spiral_galaxies = []
    edge_galaxies = []
    for f in pathlib.Path("../elliptical").glob("*.jpg"):
        filename = str(f)
        elliptical_galaxies.append(cv2.cvtColor(cv2.resize(cv2.imread(filename), (64, 64)), cv2.COLOR_BGR2RGB))
    for f in pathlib.Path("../spiral").glob("*.jpg"):
        filename = str(f)
        spiral_galaxies.append(cv2.cvtColor(cv2.resize(cv2.imread(filename), (64, 64)), cv2.COLOR_BGR2RGB))
    for f in pathlib.Path("../edge").glob("*.jpg"):
        filename = str(f)
        edge_galaxies.append(cv2.cvtColor(cv2.resize(cv2.imread(filename), (64, 64)), cv2.COLOR_BGR2RGB))
    n_ell = len(elliptical_galaxies)
    n_sp = len(spiral_galaxies)
    n_edge = len(edge_galaxies)
    print(f"{n_ell} elliptical, {n_sp} spiral and {n_edge} edge galaxies")
    set_size0 = 8 * max(n_ell, n_sp, n_edge, 10000)
    for i in range(n_ell, set_size0):
        m = cv2.getRotationMatrix2D((random.randint(12, 52), random.randint(12, 52)), random.randint(-180, 180), 1)
        elliptical_galaxies.append(cv2.warpAffine(elliptical_galaxies[random.randint(0, n_ell)], m, (64, 64)))
    for i in range(n_sp, set_size0):
        m = cv2.getRotationMatrix2D((random.randint(12, 52), random.randint(12, 52)), random.randint(-180, 180), 1)
        spiral_galaxies.append(cv2.warpAffine(spiral_galaxies[random.randint(0, n_sp)], m, (64, 64)))
    for i in range(n_edge, set_size0):
        m = cv2.getRotationMatrix2D((random.randint(12, 52), random.randint(12, 52)), random.randint(-180, 180), 1)
        edge_galaxies.append(cv2.warpAffine(edge_galaxies[random.randint(0, n_edge)], m, (64, 64)))
    galaxies = elliptical_galaxies + spiral_galaxies + edge_galaxies
    elliptical_galaxies.clear()
    spiral_galaxies.clear()
    edge_galaxies.clear()
    print(f"size of dataset: {len(galaxies)}")
    labels = []
    for i in range(set_size0):
        labels.append(numpy.array([1, 0, 0]))
    for i in range(set_size0, 2 * set_size0):
        labels.append(numpy.array([0, 1, 0]))
    for i in range(2 * set_size0, 3 * set_size0):
        labels.append(numpy.array([0, 0, 1]))
    g_array = numpy.array(galaxies)
    galaxies.clear()
    l_array = numpy.array(labels)
    labels.clear()
    return g_array, l_array


parser = argparse.ArgumentParser()
parser.add_argument("--make_galaxy_list", action="store_true")
parser.add_argument("--make_set", action="store_true")
parser.add_argument("--download_images", action="store_true")
parser.add_argument("--train_model", action="store_true")
parser.add_argument("--predict", action="store", default="")
args = parser.parse_args()
if args.make_galaxy_list:
    select_galaxies()
    exit()
if args.make_set:
    filter_galaxies(15)
    exit()
if args.download_images:
    download_images()
    exit()
if args.train_model:
    galaxies_array, labels_array = make_dataset()
    data_train, data_test, labels_train, labels_test = model_selection.train_test_split(galaxies_array, labels_array, test_size=0.2)
    model = make_model()
    model.fit(data_train, labels_train, validation_data=(data_test, labels_test), epochs=5, batch_size=32)
    model.save("classifier.h5")
    exit()
if args.predict != "":
    model = models.load_model("classifier.h5")
    img = cv2.cvtColor(cv2.resize(cv2.imread(args.predict), (64, 64)), cv2.COLOR_BGR2RGB)
    print(img.shape)
    print(model.predict(numpy.array([img])))
