import json
import os

SCAN_LIST_PATHS = {"train": os.path.join("../static/scan_lists/v0.1.0/v0.1.0_standard_splits/train.txt"),
                   "val": os.path.join("../static/scan_lists/v0.1.0/v0.1.0_standard_splits/val.txt"),
                   "test": os.path.join("../static/scan_lists/v0.1.0/v0.1.0_standard_splits/test.txt")}
DATASET_JSON_PATH = "../datasets/roosts_v0.1.0/roosts_v0.1.0.json"
OUTPUT_DIR = "/scratch2/wenlongzhao/roosts2021_ui_data/roosts_v0.1.0/annotations"

scans = []
for scan_list in SCAN_LIST_PATHS:
    scans.extend([scan.strip() for scan in open(SCAN_LIST_PATHS[scan_list], "r").readlines()])
scans = set(scans)

with open(DATASET_JSON_PATH, "r") as f:
    dataset = json.load(f)
    max_y = dataset["info"]["array_shape"][2] - 1

os.makedirs(OUTPUT_DIR, exist_ok=True)
outputs = {}

for annotation in dataset["annotations"]:
    if dataset["scans"][annotation["scan_id"]]["key"] in scans:
        track_id = annotation["sequence_id"]
        filename = dataset["scans"][annotation["scan_id"]]["key"]
        from_sunrise = dataset["scans"][annotation["scan_id"]]["minutes_from_sunrise"]
        x = annotation["x_im"]
        y = max_y - annotation["y_im"]
        r = annotation["r_im"]
        lon = annotation["x"]
        lat = -annotation["y"]
        radius = annotation["r"]

        station_year = filename[:8]
        if station_year not in outputs:
            outputs[station_year] = ["track_id,filename,from_sunrise,det_score,x,y,r,lon,lat,radius\n"]
        outputs[station_year].append(f"{track_id},{filename},{from_sunrise},1.000,{x},{y},{r},{lon},{lat},{radius}\n")

for station_year in outputs:
    with open(os.path.join(OUTPUT_DIR, station_year+"_boxes.txt"), "w") as f:
        f.writelines(outputs[station_year])