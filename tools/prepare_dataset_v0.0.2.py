"""
This script runs the data preparation pipeline to create dataset v0.0.2.
It can be modified to create customized datasets.

In most cases, changing values for VARIABLEs in Step 1 suffices.
Use None, empty lists, empty strings, etc to fill fields that are not applicable.

Step 2 involves a number of assertions. If an error is reported, see in-line comments for potential problems.
Common reasons for assertion errors and solutions include:
(1) OVERWRITE_DATASET=FALSE but the dataset version already exists
Solution: change DATASET_VERSION or delete the previous version under ../datasets.
(2) ARRAY_RENDER_CONFIG and DUALPOL_RENDER_CONFIG conflict with previous configs of same ARRAY_VERSION
Solution: change ARRAY_VERSION or clean ../static/arrays including previous_version.json.

"""

import os
import json
import numpy as np
import scipy.io as sio
import wsrlib
from wsrdata.download_radar_scans import download_by_scan_list
from wsrdata.render_npy_arrays import render_by_scan_list
from wsrdata.utils.bbox_utils import scale_XYWH_box

############### Step 1: define metadata ###############
PRETTY_PRINT_INDENT = 4 # default None; if integer n, generated json will be human-readable with n indentations

DESCRIPTION         = "The wsrdata roost dataset v0.0.2 with bounding box annotations."
COMMENTS            = "(1) There is no restrictions on radar scans and thus we use Public Domain Mark for them; " \
                      "we use the Apache License 2.0 for this dataset. " \
                      "(2) This dataset includes a) scans and annotations from v0.0.1 and b) scans from 12 " \
                      "great lakes stations and their annotations which are ecologist-screened system predictions. " \
                      "(3) Bounding boxes from dataset v0.0.1 are standardized to the heuristic scaling factor of " \
                      "0.7429 using scaling factors learned by the EM algorithm as in Cheng et al. (2019); " \
                      "the screened system predictions are not scaled. " \
                      "(4) Paths in this json use / instead of \\; this may need to be changes for a different OS."
URL                 = ""
PRE_DATASET_VERSION = {"v0.0.1": ("../datasets/roosts_v0.0.1/roosts_v0.0.1.json",
                                  "../datasets/roosts_v0.0.1/roosts_v0.0.1_standard_splits.json")}
                            # Load scans and annotations from some previous dataset version(s) to begin with
                            # Always include these previous dataset versions when running this script for
                            # the first time, to make sure all later created splits are subsets.
DATASET_VERSION     = "v0.0.2" # There can be different train/val/test splits of the dataset
SPLIT_VERSION       = "v0.0.2_standard_splits" # can consists of some v0.0.2 splits and some v0.0.1 split
INPUT_SPLIT_VERSION = "v0.0.2_standard_splits"
ANNOTATION_VERSION  = "v2.0.0" # optional -- an empty string indicates a dataset without annotations
DATE_CREATED        = "2021/12/1"
SCAN_LICENSE        = {"url": "https://creativecommons.org/share-your-work/public-domain/pdm/",
                       "name": "Public Domain Mark"}
DATASET_LICENSE     = {"url": "http://www.apache.org/licenses/",
                       "name": "Apache License 2.0"}
CATEGORIES          = ["roost"]
SUBCATEGORIES       = {"roost": ["swallow-roost", "weather-roost", "unknown-noise-roost", "AP-roost", "bad-track"]}
DEFAULT_CAT_ID      = 0 # by default, annotations are for CATEGORIES[0] which is "roost" in this template
OVERWRITE_DATASET   = False # default False; whether to overwrite the previous json file for annotations (if it exists)
OVERWRITE_SPLITS    = False # default False; whether to overwrite the previous json file for splits (if it exists)
SKIP_DOWNLOADING    = True # default True; whether to skip all downloading
SKIP_RENDERING      = True # default True; whether to skip all rendering
FORCE_RENDERING     = False # default False; whether to rerender even if an array npz already exists

SCAN_LIST_PATH      = os.path.join("../static/scan_lists", DATASET_VERSION, "scan_list.txt")
SPLIT_PATHS         = {"train": os.path.join("../static/scan_lists", DATASET_VERSION, INPUT_SPLIT_VERSION, "train.txt"),
                       # "val": os.path.join("../static/scan_lists", DATASET_VERSION, INPUT_SPLIT_VERSION, "val.txt"),
                       "test": os.path.join("../static/scan_lists", DATASET_VERSION, INPUT_SPLIT_VERSION, "test.txt")}

ARRAY_VERSION       = "v0.2.0" # corresponding to arrays defined by the following lines
ARRAY_Y_DIRECTION   = "xy" # default radar direction, y is first dim (row), large y is north, row 0 is south
ARRAY_R_MAX         = 150000.0
ARRAY_DIM           = 600
ARRAY_ATTRIBUTES    = ["reflectivity", "velocity", "spectrum_width"]
ARRAY_ELEVATIONS    = [0.5, 1.5, 2.5, 3.5, 4.5]
ARRAY_RENDER_CONFIG = {"ydirection":          ARRAY_Y_DIRECTION,
                       "fields":              ARRAY_ATTRIBUTES,
                       "coords":              "cartesian",
                       "r_min":               2125.0,       # default: first range bin of WSR-88D
                       "r_max":               ARRAY_R_MAX,  # 459875.0 default: last range bin
                       "r_res":               250,          # default: super-res gate spacing
                       "az_res":              0.5,          # default: super-res azimuth resolution
                       "dim":                 ARRAY_DIM,    # num pixels on a side in Cartesian rendering
                       "sweeps":              None,
                       "elevs":               ARRAY_ELEVATIONS,
                       "use_ground_range":    True,
                       "interp_method":       'nearest'}
DUALPOL_DIM             = 600
DUALPOL_ATTRIBUTES      = ["differential_reflectivity", "cross_correlation_ratio", "differential_phase"]
DUALPOL_ELEVATIONS      = [0.5, 1.5, 2.5, 3.5, 4.5]
DUALPOL_RENDER_CONFIG   = {"ydirection":          ARRAY_Y_DIRECTION,
                           "fields":              DUALPOL_ATTRIBUTES,
                           "coords":              "cartesian",
                           "r_min":               2125.0,       # default: first range bin of WSR-88D
                           "r_max":               ARRAY_R_MAX,  # default 459875.0: last range bin
                           "r_res":               250,          # default: super-res gate spacing
                           "az_res":              0.5,          # default: super-res azimuth resolution
                           "dim":                 DUALPOL_DIM,  # num pixels on a side in Cartesian rendering
                           "sweeps":              None,
                           "elevs":               DUALPOL_ELEVATIONS,
                           "use_ground_range":    True,
                           "interp_method":       "nearest"}

# in most cases, no need to change the following
SCAN_ROOT_DIR               = "../static/scans"
SCAN_DIR                    = os.path.join(SCAN_ROOT_DIR, "scans")
SCAN_LOG_DIR                = os.path.join(SCAN_ROOT_DIR, "logs")
SCAN_LOG_NOT_S3_DIR         = os.path.join(SCAN_ROOT_DIR, "not_s3_logs")
SCAN_LOG_ERROR_SCANS_DIR    = os.path.join(SCAN_ROOT_DIR, "error_scan_logs")
ARRAY_DIR                   = os.path.join(os.getcwd(), "../static/arrays", ARRAY_VERSION)
ARRAY_DIMENSION_ORDER       = ["field", "elevation", "y", "x"]
ARRAY_SHAPE                 = (len(ARRAY_ATTRIBUTES), len(ARRAY_ELEVATIONS), ARRAY_DIM, ARRAY_DIM)
DUALPOL_SHAPE               = (len(DUALPOL_ATTRIBUTES), len(DUALPOL_ELEVATIONS), DUALPOL_DIM, DUALPOL_DIM)
ANNOTATION_DIR              = os.path.join("../static/annotations", ANNOTATION_VERSION) if ANNOTATION_VERSION else ""
BBOX_MODE                   = "XYWH"
DATASET_DIR                 = f"../datasets/roosts_{DATASET_VERSION}"


############### Step 2: check for conflicts, update logs, create directories ###############
# make sure DATASET_VERSION is not empty, check its previous existence, create a directory for it
assert DATASET_VERSION
if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)
else:
    if os.path.exists(f"{DATASET_DIR}/roosts_{DATASET_VERSION}.json") and OVERWRITE_DATASET:
        print("Will overwrite the existing json file for dataset definition. "
              "Please manually clean or overwrite old visualization data as needed.")
    if os.path.exists(f"{DATASET_DIR}/roosts_{SPLIT_VERSION}.json") and OVERWRITE_SPLITS:
        print("Will overwrite the existing json file for splits.")

# make sure PRE_DATASET_VERSION, if specified, exists, and has no conflict with the current dataset
for pre_dataset_version, pre_dataset_json_paths in PRE_DATASET_VERSION.items():
    assert os.path.exists(pre_dataset_json_paths[0])
    assert os.path.exists(pre_dataset_json_paths[1])
    pre_dataset = json.load(open(pre_dataset_json_paths[0], 'r'))
    assert pre_dataset["categories"] == CATEGORIES

# make sure SCAN_LIST_PATH and SPLIT_PATHS exist
assert os.path.exists(SCAN_LIST_PATH)
for split in SPLIT_PATHS: assert os.path.exists(SPLIT_PATHS[split])
# make sure ANNOTATION_DIR exists when ANNOTATION_VERSION is not empty
if ANNOTATION_VERSION: assert os.path.exists(ANNOTATION_DIR)

# if we download or render, make sure SCAN and ARRAY related paths are ready
if not SKIP_DOWNLOADING or not SKIP_RENDERING:
    if not os.path.exists(SCAN_ROOT_DIR): os.mkdir(SCAN_ROOT_DIR)
    if not os.path.exists(SCAN_DIR): os.mkdir(SCAN_DIR)
    if not os.path.exists(SCAN_LOG_DIR): os.mkdir(SCAN_LOG_DIR)
    if not os.path.exists(SCAN_LOG_NOT_S3_DIR): os.mkdir(SCAN_LOG_NOT_S3_DIR)
    if not os.path.exists(SCAN_LOG_ERROR_SCANS_DIR): os.mkdir(SCAN_LOG_ERROR_SCANS_DIR)

    # make sure ARRAY_VERSION is not empty and does not conflict with existing versions
    assert ARRAY_VERSION
    if not os.path.exists("../static/arrays"): os.mkdir("../static/arrays")
    existing_versions = os.listdir("../static/arrays") # those currently in the directory
    previous_versions = {} # those previously recorded at some point
    if os.path.exists("../static/arrays/previous_versions.json"):
        with open("../static/arrays/previous_versions.json", "r") as f:
            previous_versions = json.load(f)
    # make sure existing versions are all recorded in previous_versions.json
    # if so, we use previous_versions.json as a reference to detect version conflicts
    # otherwise, manual cleaning is required
    for v in existing_versions:
        if v != "previous_versions.json" and v != ".gitignore":
            assert v in previous_versions
    # make sure there is no config conflict
    # otherwise, either choose a new ARRAY_VERSION or clean the existing/previous version
    if ARRAY_VERSION in previous_versions:
        assert previous_versions[ARRAY_VERSION] == {"array": ARRAY_RENDER_CONFIG, "dualpol": DUALPOL_RENDER_CONFIG}
    # initiate ARRAY_VERSION as a new version
    else:
        previous_versions[ARRAY_VERSION] = {"array": ARRAY_RENDER_CONFIG, "dualpol": DUALPOL_RENDER_CONFIG}
        with open("../static/arrays/previous_versions.json", "w") as f:
            json.dump(previous_versions, f, indent=PRETTY_PRINT_INDENT)
    if not os.path.exists(ARRAY_DIR): os.mkdir(ARRAY_DIR)


############### Step 3: sketch the dataset definition ###############
info = {
    "description":              DESCRIPTION,
    "comments":                 COMMENTS,
    "url":                      URL,
    "dataset_version":          DATASET_VERSION,
    "license":                  DATASET_LICENSE,
    "scan_license":             SCAN_LICENSE,
    "annotation_version":       ANNOTATION_VERSION,
    "date_created":             DATE_CREATED,
    "array_version":            ARRAY_VERSION,
    "array_dir":                ARRAY_DIR,
    "array_dimension_order":    ARRAY_DIMENSION_ORDER,
    "array_ydirection":         ARRAY_Y_DIRECTION,
    "array_shape":              ARRAY_SHAPE,
    "array_fields":             ARRAY_ATTRIBUTES,
    "array_elevations":         ARRAY_ELEVATIONS,
    "array_r_max":              ARRAY_R_MAX,
    "dualpol_shape":            DUALPOL_SHAPE,
    "dualpol_fields":           DUALPOL_ATTRIBUTES,
    "dualpol_elevations":       DUALPOL_ELEVATIONS,
    "bbox_mode":                BBOX_MODE,
}

dataset = {
    "info":                 info,
    "scans":                [],
    "annotations":          [],
    "categories":           CATEGORIES,
    "subcategories":        SUBCATEGORIES,
}

for pre_dataset_version, pre_dataset_json_paths in PRE_DATASET_VERSION.items():
    pre_dataset = json.load(open(pre_dataset_json_paths[0], 'r'))
    if "pre_dataset_info" not in dataset["info"]:
        dataset["info"]["pre_dataset_info"] = []
    dataset["info"]["pre_dataset_info"].append(pre_dataset["info"])


############### Step 4: Download radar scans ###############
if not SKIP_DOWNLOADING:
    print("Downloading scans...")
    download_errors = download_by_scan_list(
        SCAN_LIST_PATH, SCAN_DIR,
        os.path.join(SCAN_LOG_DIR, f"{DATASET_VERSION}.log"),
        os.path.join(SCAN_LOG_NOT_S3_DIR, f"{DATASET_VERSION}.log"),
        os.path.join(SCAN_LOG_ERROR_SCANS_DIR, f"{DATASET_VERSION}.log")
    )


############### Step 5: Render arrays from radar scans ###############
if not SKIP_RENDERING:
    print("Rendering arrays...")
    array_errors, dualpol_errors = render_by_scan_list(
        SCAN_LIST_PATH, SCAN_DIR, ARRAY_DIR,
        ARRAY_RENDER_CONFIG, DUALPOL_RENDER_CONFIG, FORCE_RENDERING
    )


############### Step 6: Populate the dataset definition and save to json ###############
create_annotation_json = False
if not os.path.exists(f"{DATASET_DIR}/roosts_{DATASET_VERSION}.json") or OVERWRITE_DATASET:
    create_annotation_json = True

    # Preparation: load annotations into a dictionary where scan names are keys
    annotation_dict = {}
    minutes_from_sunrise_dict = {}
    if ANNOTATION_VERSION:
        print("Loading annotations....")

        # from previous dataset versions
        for pre_dataset_version, pre_dataset_json_paths in PRE_DATASET_VERSION.items():
            pre_dataset = json.load(open(pre_dataset_json_paths[0], 'r'))
            if pre_dataset["info"]["dataset_version"] in ["v0.0.1", "v0.1.0"]:
                for annotation in pre_dataset["annotations"]:
                    new_annotation = {
                        "id":               None,  # temporarily set to None
                        "scan_id":          None,  # temporarily set to None
                        "category_id":      annotation["category_id"],
                        "dataset_version":  pre_dataset["info"]["dataset_version"],
                        "track_id":         annotation["sequence_id"],
                        "lon":              None,
                        "lat":              None,
                        "radius":           None,
                        "x_im":             annotation["x_im"],
                        "y_im":             annotation["y_im"],
                        "r_im":             annotation["r_im"],
                        "bbox":             annotation["bbox"],
                        "bbox_area":        annotation["bbox_area"],
                        "subcategory":      None,
                        "notes":            None,
                        "day_notes":        None,
                    }
                    if pre_dataset["scans"][annotation["scan_id"]]["key"] in annotation_dict:
                        annotation_dict[pre_dataset["scans"][annotation["scan_id"]]["key"]].append(new_annotation)
                    else:
                        annotation_dict[pre_dataset["scans"][annotation["scan_id"]]["key"]] = [new_annotation]
            else:
                for annotation in pre_dataset["annotations"]:
                    annotation["id"] = None
                    annotation["scan_id"] = None
                    if pre_dataset["scans"][annotation["scan_id"]]["key"] in annotation_dict:
                        annotation_dict[pre_dataset["scans"][annotation["scan_id"]]["key"]].append(annotation)
                    else:
                        annotation_dict[pre_dataset["scans"][annotation["scan_id"]]["key"]] = [annotation]

        # new to this dataset version
        csv_files = sorted(os.listdir(os.path.join(ANNOTATION_DIR, "csv")))
        csv_files = [os.path.join(ANNOTATION_DIR, "csv", f) for f in csv_files if f.endswith("csv")]
        for csv_file in csv_files:
            # fields are:
            #       0 track_id, 1 filename, 2 from_sunrise, 3 det_score, 4 x, 5 y, 6 r, 7 lon, 8 lat, 9 radius,
            #       10 local_time, 11 station, 12 date, 13 time, 14 local_date, 15 length,
            #       16 tot_score, 17 avg_score, 18 viewed, 19 user_labeled, 20 label, 21 original_label,
            #       22 notes: 'LARGE', 'nr', 'long', 'large', 'rn', 'shrinks', 'shrink',
            #       23 day_notes: 'pap', 'psp', 'weather', '2', 'ap', 'AP', 'miss', 'cluster', 'clusters'.
            annotations = [annotation.strip().split(",") for annotation in open(csv_file, "r").readlines()[1:]]
            for annotation in annotations:
                # skip if not ecologist-verified or verified to be non-roost or duplicate
                if annotation[18].lower() != 'true' or annotation[20] not in SUBCATEGORIES[CATEGORIES[DEFAULT_CAT_ID]]:
                    continue

                x_im = float(annotation[4])
                y_im = ARRAY_DIM - float(annotation[5]) # from image direction to geographic direction
                r_im = float(annotation[6])
                bbox = [int(x_im - r_im), int(y_im - r_im),
                        int(x_im + r_im) - int(x_im - r_im),
                        int(y_im + r_im) - int(y_im - r_im)]
                new_annotation = {
                    "id":               None, # temporarily set to None
                    "scan_id":          None, # temporarily set to None
                    "category_id":      DEFAULT_CAT_ID,
                    "dataset_version":  DATASET_VERSION,
                    "track_id":         annotation[0],
                    "lon":              float(annotation[7]),
                    "lat":              float(annotation[8]),
                    "radius":           float(annotation[9]),
                    "x_im":             x_im,
                    "y_im":             y_im,
                    "r_im":             r_im,
                    "bbox":             bbox,
                    "bbox_area":        bbox[2] * bbox[3],
                    "subcategory":      annotation[20],
                    "notes":            annotation[22],
                    "day_notes":        annotation[23],
                }
                if annotation[1] in annotation_dict:
                    annotation_dict[annotation[1]].append(new_annotation)
                else:
                    annotation_dict[annotation[1]] = [new_annotation]
                minutes_from_sunrise_dict[annotation[1]] = int(annotation[2])

    # Load scan names and populate the dataset definition
    print("Populating the dataset definition...")
    scan_id = 0
    annotation_id = 0

    # from previous dataset versions
    for pre_dataset_version, pre_dataset_json_paths in PRE_DATASET_VERSION.items():
        pre_dataset = json.load(open(pre_dataset_json_paths[0], 'r'))
        for scan in pre_dataset["scans"]:
            dataset["scans"].append({
                "id":                   scan_id,
                "annotation_ids":       [],
                "dataset_version":      pre_dataset["info"]["dataset_version"],
                "key":                  scan["key"],
                "minutes_from_sunrise": scan["minutes_from_sunrise"],
                "array_path":           scan["array_path"],
            })

            if scan["key"] in annotation_dict:
                for annotation in annotation_dict[scan["key"]]:
                    annotation["id"] = annotation_id
                    dataset["scans"][scan_id]["annotation_ids"].append(annotation_id)
                    annotation_id += 1
                    annotation["scan_id"] = scan_id
                    dataset["annotations"].append(annotation)

            scan_id += 1

    # new to this dataset version
    scans = [scan.strip() for scan in open(SCAN_LIST_PATH, "r").readlines()]
    for n, key in enumerate(scans):
        # add array to dataset
        dataset["scans"].append({
            "id":                   scan_id,
            "annotation_ids":       [],
            "dataset_version":      DATASET_VERSION,
            "key":                  key,
            "minutes_from_sunrise": minutes_from_sunrise_dict[key] if key in minutes_from_sunrise_dict else None,
            "array_path":           f"{key[4:8]}/{key[8:10]}/{key[10:12]}/{key[0:4]}/{key}.npz",
        })

        if key in annotation_dict:
            for annotation in annotation_dict[key]:
                annotation["id"] = annotation_id
                dataset["scans"][scan_id]["annotation_ids"].append(annotation_id)
                annotation_id += 1
                annotation["scan_id"] = scan_id
                dataset["annotations"].append(annotation)

        scan_id += 1

    print("Saving the dataset definition to json...")
    with open(f"{DATASET_DIR}/roosts_{DATASET_VERSION}.json", 'w') as f:
        json.dump(dataset, f, indent=PRETTY_PRINT_INDENT)


############### Step 7: Save a set of splits to json ###############
if not os.path.exists(f"{DATASET_DIR}/roosts_{SPLIT_VERSION}.json") or OVERWRITE_SPLITS:
    if not create_annotation_json:
        dataset = json.load(open(f"{DATASET_DIR}/roosts_{DATASET_VERSION}.json", 'r'))
    scan_key_to_scan_id = {}
    for scan in dataset["scans"]:
        scan_key_to_scan_id[scan["key"]] = scan["id"]

    print("Saving the dataset splits...")
    splits = {}
    for split in SPLIT_PATHS:
        splits[split] = []

    # from previous dataset versions
    for pre_dataset_version, pre_dataset_json_paths in PRE_DATASET_VERSION.items():
        pre_dataset = json.load(open(pre_dataset_json_paths[0], 'r'))
        pre_dataset_splits = json.load(open(pre_dataset_json_paths[1], 'r'))
        for split in pre_dataset_splits:
            splits[split].extend([
                scan_key_to_scan_id[pre_dataset["scans"][scan_id]["key"]] for scan_id in pre_dataset_splits[split]
            ])

    # new to this dataset version
    for split in SPLIT_PATHS:
        splits[split].extend([
            scan_key_to_scan_id[scan.strip()] for scan in open(SPLIT_PATHS[split], "r").readlines()
            if scan.strip() in scan_key_to_scan_id
        ])

    with open(f"{DATASET_DIR}/roosts_{SPLIT_VERSION}.json", 'w') as f:
        json.dump(splits, f, indent=PRETTY_PRINT_INDENT)


print("All done.")