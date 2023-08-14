import os
import json
import numpy as np

# Load datasets which include metadata, all scans, all annotations
dataset = json.load(open(f"../datasets/roosts_v0.2.0/roosts_v0.2.0.json", 'r'))
scan_key_to_scan_id = {}
for scan in dataset["scans"]:
    scan_key_to_scan_id[scan["key"]] = scan["id"]

# Get id of legacy training data
pre_dataset = json.load(open("../datasets/roosts_v0.1.0/roosts_v0.1.0.json", 'r'))
pre_dataset_splits = json.load(open("../datasets/roosts_v0.1.0/roosts_v0.1.0_standard_splits.json", 'r'))
legacy_train_scans = [
    scan_key_to_scan_id[pre_dataset["scans"][scan_id]["key"]]
    for scan_id in pre_dataset_splits["train"]
]

# Produce new splits
INPUT_DIR = "../static/scan_lists/v0.2.10"
OUTPUT_DIR = "../datasets/roosts_v0.2.10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_scans(input_file):
    scan_list = []
    for scan in open(input_file, "r").readlines():
        assert scan.strip() in scan_key_to_scan_id
        scan_list.append(scan_key_to_scan_id[scan.strip()])
    return scan_list

def save_to_json(scan_list, output_file):
    with open(output_file, 'w') as f:
        json.dump(sorted(scan_list), f)

for station in ['KGRR', 'KIWX', 'KLOT', 'KMKX']:
    # same station-year experiments
    for year in ['2005', '2010', '2015', '2020']:
        # validation
        scan_list = read_scans(f"{INPUT_DIR}/v0.2.10_{station}_{year}_splits/valid.txt")
        save_to_json(scan_list, f"{OUTPUT_DIR}/{station}_{year}_valid.json")
        print(f"{len(scan_list)}\t{station}_{year}_valid")

        # testing
        scan_list = read_scans(f"{INPUT_DIR}/v0.2.10_{station}_{year}_splits/test.txt")
        save_to_json(scan_list, f"{OUTPUT_DIR}/{station}_{year}_test.json")
        print(f"{len(scan_list)}\t{station}_{year}_test")

        # training
        for ratio in ['0.0625', '0.125', '0.25', '0.5']:
            # only new
            scan_list = read_scans(f"{INPUT_DIR}/v0.2.10_{station}_{year}_splits/train_{ratio}.txt")
            save_to_json(scan_list, f"{OUTPUT_DIR}/{station}_{year}_train_{ratio}.json")
            print(f"{len(scan_list)}\t{station}_{year}_train_{ratio}")

            # union
            scan_list.extend(legacy_train_scans)
            save_to_json(scan_list, f"{OUTPUT_DIR}/{station}_{year}_train_{ratio}_union.json")
            print(f"{len(scan_list)}\t{station}_{year}_train_{ratio}_union")

    # same station other year experiments
    for (years, ratio) in [
        ('2016', '0.5'), ('2017', '0.5'), ('2018', '0.5'), ('2019', '0.5'),
        ('2016-2020', '0.25'), ('2018-2020', '0.25'), ('2019-2020', '0.25'),
        ('2014-2016-2018-2020', '0.125'), ('2017-2018-2019-2020', '0.125'),
    ]:
        scan_list = []
        for year in years.split('-'):
            scan_list.extend(read_scans(f"{INPUT_DIR}/v0.2.10_{station}_{year}_splits/train_{ratio}.txt"))
        save_to_json(scan_list, f"{OUTPUT_DIR}/{station}_{years}_train_{ratio}.json")
        print(f"{len(scan_list)}\t{station}_{years}_train_{ratio}")

    # era experiments
    # train
    ratio = 0.5
    for year in ['2007', '2009']:
        scan_list = read_scans(f"{INPUT_DIR}/v0.2.10_{station}_{year}_splits/train_{ratio}.txt")
        save_to_json(scan_list, f"{OUTPUT_DIR}/{station}_{year}_train_{ratio}.json")
        print(f"{len(scan_list)}\t{station}_{year}_train_{ratio}")
    for year in ['2008']:
        # validation
        scan_list = read_scans(f"{INPUT_DIR}/v0.2.10_{station}_{year}_splits/valid.txt")
        save_to_json(scan_list, f"{OUTPUT_DIR}/{station}_{year}_valid.json")
        print(f"{len(scan_list)}\t{station}_{year}_valid")
        # testing
        scan_list = read_scans(f"{INPUT_DIR}/v0.2.10_{station}_{year}_splits/test.txt")
        save_to_json(scan_list, f"{OUTPUT_DIR}/{station}_{year}_test.json")
        print(f"{len(scan_list)}\t{station}_{year}_test")
