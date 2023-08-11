"""
Download and render for scans listed in SCAN_LIST_PATH
Launch this scripts multiple times in parallel for accelerated dataset creation
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
PRETTY_PRINT_INDENT = None # default None; if integer n, generated json will be human-readable with n indentations

DESCRIPTION         = "The wsrdata roost dataset v0.1.0 with bbox annotations."
COMMENTS            = "(1) There is no restrictions on radar scans and thus we use Public Domain Mark for them; " \
                      "we use the Apache License 2.0 for this dataset. " \
                      "(2) Bounding boxes are standardized to the heuristic scaling factor of 0.7429 " \
                      "using scaling factors learned by the EM algorithm as in Cheng et al. (2019). " \
                      "(3) Paths in this json use / instead of \\; this may need to be changes for a different OS."
URL                 = ""
DATASET_VERSION     = "v0.1.0" # There can be different train/val/test splits of the dataset denoted as v0.0.1_xxx.
SPLIT_VERSION       = "v0.1.0_ordered_splits"
ANNOTATION_VERSION  = "v1.0.0" # optional -- an empty string indicates a dataset without annotations
USER_MODEL_VERSION  = "v1.0.0_hardEM200000"
DATE_CREATED        = "2021/04/20"
SCAN_LICENSE        = {"url": "https://creativecommons.org/share-your-work/public-domain/pdm/",
                       "name": "Public Domain Mark"}
DATASET_LICENSE     = {"url": "http://www.apache.org/licenses/",
                       "name": "Apache License 2.0"}
CATEGORIES          = ["roost"]
DEFAULT_CAT_ID      = 0 # by default, annotations are for CATEGORIES[0] which is "roost" in this template
OVERWRITE_DATASET   = False # default False; whether to overwrite the previous json file for annotations (if it exists)
OVERWRITE_SPLITS    = False # default False; whether to overwrite the previous json file for splits (if it exists)
SKIP_DOWNLOADING    = False # default False; whether to skip all downloading
SKIP_RENDERING      = False # default False; whether to skip all rendering
FORCE_RENDERING     = True # default False; whether to rerender even if an array npz already exists

SCAN_LIST_PATH      = "prepare_dataset_v0.1.0_help/test_2.txt"
# SCAN_LIST_PATH      = os.path.join("../static/scan_lists", DATASET_VERSION, SPLIT_VERSION, "val.txt")
SPLIT_PATHS         = {"train": os.path.join("../static/scan_lists", DATASET_VERSION, SPLIT_VERSION, "train.txt"),
                       "val": os.path.join("../static/scan_lists", DATASET_VERSION, SPLIT_VERSION, "val.txt"),
                       "test": os.path.join("../static/scan_lists", DATASET_VERSION, SPLIT_VERSION, "test.txt")}

ARRAY_VERSION       = "v0.1.0" # corresponding to arrays defined by the following lines
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

# manually imported from static/user_models/v1.0.0/hardEM200000_user_model_python2.pkl
TARGET_SCALE_FACTOR     = 0.7429 # average sheldon factor
BBOX_SCALING_FACTORS    = {'Ftian-KOKX': 0.7827008296465084,
                           'William Curran-KDOX': 0.6671858060703622,
                           'andrew-KAMX': 0.8238429277541144,
                           'andrew-KHGX': 0.8021155634196264,
                           'andrew-KJAX': 0.9397206576582352,
                           'andrew-KLCH': 0.7981654079788019,
                           'andrew-KLIX': 1.003359702917803,
                           'andrew-KMLB': 0.8846939182400024,
                           'andrew-KTBW': 1.0745160463520484,
                           'andrew-KTLH': 0.8121429842343971,
                           'anon-KDOX': 0.6393409410259764,
                           'anon-KLIX': 0.8789372720576193,
                           'anon-KTBW': 0.8777182885471609,
                           'jafer1-KDOX': 0.643700604491143,
                           'jafermjj-KDOX': 0.629814055371781,
                           'jberger1-KAMX': 1.0116521039423771,
                           'jberger1-KLIX': 0.9350564477085113,
                           'jberger1-KMLB': 1.01208151592683,
                           'jberger1-KTBW': 1.0710975633513655,
                           'jpodrat-KLIX': 1.0258838999961304,
                           'sheldon-KAMX': 1.0190194757755286,
                           'sheldon-KDOX': 0.6469252517936639,
                           'sheldon-KLIX': 0.7086575697533594,
                           'sheldon-KMLB': 0.8441916918113227,
                           'sheldon-KOKX': 0.6049163038774339,
                           'sheldon-KRTX': 0.5936236006148872,
                           'sheldon-KTBW': 0.7830289430054851,}

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

# make sure SCAN_LIST_PATH, SCAN_ROOT_DIR, etc exist
assert os.path.exists(SCAN_LIST_PATH)
for split in SPLIT_PATHS: assert os.path.exists(SPLIT_PATHS[split])
if not os.path.exists(SCAN_ROOT_DIR): os.mkdir(SCAN_ROOT_DIR)
if not os.path.exists(SCAN_DIR): os.mkdir(SCAN_DIR)
if not os.path.exists(SCAN_LOG_DIR): os.mkdir(SCAN_LOG_DIR)
if not os.path.exists(SCAN_LOG_NOT_S3_DIR): os.mkdir(SCAN_LOG_NOT_S3_DIR)
if not os.path.exists(SCAN_LOG_ERROR_SCANS_DIR): os.mkdir(SCAN_LOG_ERROR_SCANS_DIR)
# make sure ANNOTATION_DIR exists when ANNOTATION_VERSION is not empty
if ANNOTATION_VERSION: assert os.path.exists(ANNOTATION_DIR)

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

