import os
import numpy as np
import json
from wsrlib import pyart, radar2mat
from wsrdata.utils.bbox_utils import scale_XYWH_box
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
OUTPUT_DIR = None # if not None, save all figures directly under this
OUTPUT_ROOT = None # if not None, create subdirectories {year}/{month}/{date}/{station} to save figures

# define which channels for which scans to visualize, which json to be the source of annotations
CHANNELS = [("reflectivity", 0.5), ("reflectivity", 1.5), ("velocity", 0.5)]
# (1) check if visualization works properly for the toy dataset version
SCAN_LIST_PATHS = {"all": os.path.join("../static/scan_lists/v0.0.1/scan_list.txt")}
JSON_PATH = "../datasets/roosts_v0.0.1/roosts_v0.0.1.json"
OUTPUT_DIR = "../datasets/roosts_v0.0.1/visualization"
# (2) check if sampled boxes for each annotator-station pair makes sense
# SCAN_LIST_PATHS = {pair: f"../static/scan_lists/v0.1.0/v0.1.0_subset_for_debugging/{pair}.txt" for pair in
#                ['Ftian-KOKX', 'William Curran-KDOX', 'andrew-KAMX', 'andrew-KHGX', 'andrew-KJAX',
#                 'andrew-KLCH', 'andrew-KLIX', 'andrew-KMLB', 'andrew-KTBW', 'andrew-KTLH',
#                 'anon-KDOX', 'anon-KLIX', 'anon-KTBW', 'jafer1-KDOX', 'jafermjj-KDOX',
#                 'jberger1-KAMX', 'jberger1-KLIX', 'jberger1-KMLB', 'jberger1-KTBW', 'jpodrat-KLIX',
#                 'sheldon-KAMX', 'sheldon-KDOX', 'sheldon-KLIX', 'sheldon-KMLB', 'sheldon-KOKX',
#                 'sheldon-KRTX', 'sheldon-KTBW']
#                    if os.path.exists(f"../static/scan_lists/v0.1.0/v0.1.0_subset_for_debugging/{pair}.txt")}
# JSON_PATH = "../datasets/roosts_v0.1.0/roosts_v0.1.0.json"
# OUTPUT_DIR = "../datasets/roosts_v0.1.0/visualization"

# visualization settings
COLOR_ARRAY = [
    '#006400', # for not scaled boxes
    '#FF00FF', # scaled to RCNN then to user factor 0.7429 which is sheldon average
    '#800080',
    '#FFA500',
    '#FFFF00'
]
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
        print(f"Processing the {n+1}th scan")
        scan = dataset["scans"][scan_to_id[SCAN]]
        array = np.load(os.path.join(dataset["info"]["array_dir"], scan["array_path"]))["array"]

        fig, axs = plt.subplots(int(np.ceil(len(CHANNELS)/3)), 3,
                                figsize=(21, 7*int(np.ceil(len(CHANNELS)/3))),
                                constrained_layout=True)
        for i, (attr, elev) in enumerate(CHANNELS):
            subplt = axs[i]
            subplt.axis('off')
            subplt.set_title(f"{attr}, elev: {elev}", fontsize=18)
            cm = plt.get_cmap(pyart.config.get_field_colormap(attr))
            rgb = cm(NORMALIZERS[attr](array[attributes.index(attr), elevations.index(elev), :, :]))
            rgb = rgb[:, :, :3]  # omit the fourth alpha dimension, NAN are black but not white
            subplt.imshow(rgb, origin='lower')
            for annotation_id in scan["annotation_ids"]:
                bbox = dataset["annotations"][annotation_id]["bbox"]
                # uncomment the following lines if the input bboxes contain annotator biases
                # subplt.add_patch(
                #     plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                #                   fill=False,
                #                   edgecolor=COLOR_ARRAY[0],
                #                   linewidth=1.2)
                # )
                # subplt.text(10, 10, 'not scaled',
                #             bbox=dict(facecolor='white', alpha=0.5), fontsize=14, color=COLOR_ARRAY[0])
                #
                # bbox = scale_XYWH_box(bbox, dataset["info"]["bbox_scaling_factors"][annotation["bbox_annotator"]])
                subplt.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                  fill=False,
                                  edgecolor=COLOR_ARRAY[1],
                                  linewidth=1.2)
                )
                subplt.text(10, 560, 'scaled -> RCNN -> sheldon average (0.7429)',
                            bbox=dict(facecolor='white', alpha=0), fontsize=14, color=COLOR_ARRAY[1])

        # save
        if OUTPUT_DIR is not None and len(SCAN_LIST_PATHS) == 1:
            output_dir = OUTPUT_DIR
        elif OUTPUT_DIR is not None and len(SCAN_LIST_PATHS) > 1:
            output_dir = os.path.join(OUTPUT_DIR, scan_list)
        elif OUTPUT_ROOT is not None:
            station = SCAN[0:4]
            year = SCAN[4:8]
            month = SCAN[8:10]
            date = SCAN[10:12]
            output_dir = os.path.join(OUTPUT_ROOT, f"{year}/{month}/{date}/{station}")
        else:
            print("No output directory defined. "
                  "Please assign values to OUTPUT_DIR or OUTPUT_ROOT and rerun the program. ")
            exit()
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, SCAN + ".png"), bbox_inches="tight")
        plt.close(fig)