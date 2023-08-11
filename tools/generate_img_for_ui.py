import os
import numpy as np
import json
from wsrlib import pyart
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from matplotlib import image

SCAN_LIST_PATHS = {"train": os.path.join("../static/scan_lists/v0.1.0/v0.1.0_standard_splits/train.txt"),
                   "val": os.path.join("../static/scan_lists/v0.1.0/v0.1.0_standard_splits/val.txt"),
                   "test": os.path.join("../static/scan_lists/v0.1.0/v0.1.0_standard_splits/test.txt")}
JSON_PATH = "../datasets/roosts_v0.1.0/roosts_v0.1.0.json"
CHANNELS = {("reflectivity", 0.5): "/scratch2/wenlongzhao/roosts2021_ui_data/roosts_v0.1.0/ref0.5_images",
            ("velocity", 0.5): "/scratch2/wenlongzhao/roosts2021_ui_data/roosts_v0.1.0/rv0.5_images"}
for channel in CHANNELS: os.makedirs(CHANNELS[channel], exist_ok=True)

# visualization settings
NORMALIZERS = {
        'reflectivity':              pltc.Normalize(vmin=  -5, vmax= 35),
        'velocity':                  pltc.Normalize(vmin= -15, vmax= 15),
        'spectrum_width':            pltc.Normalize(vmin=   0, vmax= 10),
        'differential_reflectivity': pltc.Normalize(vmin=  -4, vmax= 8),
        'differential_phase':        pltc.Normalize(vmin=   0, vmax= 250),
        'cross_correlation_ratio':   pltc.Normalize(vmin=   0, vmax= 1.1)
}

# load data
with open(JSON_PATH, "r") as f:
    dataset = json.load(f)
scan_to_id = {}
for scan in dataset["scans"]:
    scan_to_id[scan["key"]] = scan["id"]
attributes = dataset["info"]["array_fields"]
elevations = dataset["info"]["array_elevations"]

# plot
for scan_list in SCAN_LIST_PATHS:
    scans = [scan.strip() for scan in open(SCAN_LIST_PATHS[scan_list], "r").readlines()]

    for n, SCAN in enumerate(scans):
        if n % 1000 == 0:
            print(f"Processing the {n+1}th scan")
        scan = dataset["scans"][scan_to_id[SCAN]]
        array = np.load(os.path.join(dataset["info"]["array_dir"], scan["array_path"]))["array"]

        for channel in CHANNELS:
            attr = channel[0]
            elev = channel[1]
            cm = plt.get_cmap(pyart.config.get_field_colormap(attr))
            rgb = cm(NORMALIZERS[attr](array[attributes.index(attr), elevations.index(elev), :, :]))
            rgb = rgb[::-1, :, :3]  # flip the y axis; omit the fourth alpha dimension, NAN are black but not white
            image.imsave(os.path.join(CHANNELS[channel], SCAN+".png"), rgb)
