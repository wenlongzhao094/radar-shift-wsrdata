"""
This script is a modifiable template which is written for dataset v0.2.B,
where B indicates different sampling strategies.

It reads the json generated by organize_screened_csv_as_json.py in the format of
station_years[station_year] = {
    'all_scans_with_check':     {},  # scan: {"avg_dbz": float, "dualpol": True/False}                  SCAN
    'all_days_to_scans':        {},  # day: set(scan)                                                   DAY

    'n_roost_annotations':              0,                                                              ANN
    'n_roost_annotations_not_miss_day': 0,                                                              ANN
    'n_bad_track_annotations':          0,                                                              ANN
    'scans_with_roosts':                set(),  # positive scans                                        SCAN
    'roost_days':                       set(),  # days with roosts                                      DAY

    'n_scans_without_roosts_in_roost_days': 0,  # negative scans                                        SCAN
    'n_scans_in_non_roost_days':            0,  # scans from sampled non_roost_days become negatives    SCAN
    'non_roost_days':                       set(),  # days without roosts                               DAY
}
and defines scan lists and splits under **static/scan_lists/v0.2.B**.
"""

import numpy as np
import json, os
import random

# input
STATIONS = [
    'KAPX', 'KBUF', 'KCLE', 'KDLH', 'KDTX', 'KGRB',
    'KGRR', 'KIWX', 'KLOT', 'KMKX', 'KMQT', 'KTYX',
]
station_years = {}
for station in STATIONS:
    station = json.load(open(f'prepare_dataset_v0.2.0_help/all_days_all_scans_{station}.json', 'r'))
    for station_year in sorted(list(station.keys())):
        station_years[station_year] = station[station_year]
# sampling
# N_NEG_DAYS_TO_N_POS_DAYS = 1
TEST_RATIO = 0.3
VALID_RATIO = 0.2
TRAIN_RATIOS = [0.0625, 0.125, 0.25, 0.5]
TRAIN_RATIOS = sorted(TRAIN_RATIOS)
# output
DATASET_VERSION = 'v0.2.10'
SCAN_LIST_DIR = f'../static/scan_lists/{DATASET_VERSION}'
os.makedirs(SCAN_LIST_DIR, exist_ok=True)
# ../static/scan_lists/v0.2.10/scan_list.txt -> same for v0.2
with open(os.path.join(SCAN_LIST_DIR, 'scan_list.txt'), "w") as f:
    for station_year in station_years:
        f.writelines([scan + '\n' for scan in station_years[station_year]['all_scans_with_check']])
output_splits = {}

# Create splits
# ../static/scan_lists/v0.2.10/v0.2.10_SSSS_YYYY_splits/{train_<train_ratio>,valid,test}.txt
random.seed(1)
for station_year in station_years:
    train_scans = {train_ratio: [] for train_ratio in sorted(TRAIN_RATIOS)}
    valid_scans = []
    test_scans = []

    n_days = len(station_years[station_year]['roost_days']) + len(station_years[station_year]['non_roost_days'])
    train_days = {train_ratio: [] for train_ratio in sorted(TRAIN_RATIOS)}
    valid_days = []
    test_days = []
    non_test_days = []

    # Positive test days -> 30% of positive days
    roost_days = list(station_years[station_year]['roost_days'])
    random.shuffle(roost_days)
    _n_pos_test = int(len(roost_days) * TEST_RATIO)
    test_days.extend(roost_days[:_n_pos_test])
    non_test_days.extend(roost_days[_n_pos_test:])

    # Negative test days -> the less of 30% of negative days and the same amount as positive test days
    non_roost_days = list(station_years[station_year]['non_roost_days'])
    random.shuffle(non_roost_days)
    _n_neg_test = int(len(non_roost_days) * TEST_RATIO)
    # control the ratio between positive and negative days, so the eval is not overwhelmed by neg days
    # test_days.extend(non_roost_days[:int(min(_n_neg_test, _n_pos_test * N_NEG_DAYS_TO_N_POS_DAYS))])
    test_days.extend(non_roost_days[:_n_neg_test])
    non_test_days.extend(non_roost_days[_n_neg_test:])

    # Train and validation days
    n_train_days = int(TRAIN_RATIOS[-1] * n_days)
    n_valid_days = int(VALID_RATIO * n_days)
    assert n_train_days + n_valid_days <= len(non_test_days)
    random.shuffle(non_test_days)
    for train_ratio in TRAIN_RATIOS:
        train_days[train_ratio].extend(non_test_days[:int(train_ratio * n_days)])
    valid_days.extend(non_test_days[-n_valid_days:])

    # Day to scans
    def days_to_scans(days, scans):
        for day in sorted(days):
            scans.extend(station_years[station_year]["all_days_to_scans"][day])

    for train_ratio in TRAIN_RATIOS:
        days_to_scans(train_days[train_ratio], train_scans[train_ratio])
    days_to_scans(valid_days, valid_scans)
    days_to_scans(test_days, test_scans)
    assert not set(train_scans[TRAIN_RATIOS[-1]]).intersection(set(valid_scans))
    assert not set(train_scans[TRAIN_RATIOS[-1]]).intersection(set(test_scans))
    assert not set(valid_scans).intersection(set(test_scans))

    # Save scans and stats
    splits_dir = os.path.join(SCAN_LIST_DIR, f'{DATASET_VERSION}_{station_year}_splits')
    os.makedirs(splits_dir, exist_ok=True)

    def save_split(filename, scans):
        with open(os.path.join(splits_dir, filename), "w") as f:
            f.writelines([scan + '\n' for scan in sorted(list(scans))])

    for train_ratio in TRAIN_RATIOS:
        save_split(f'train_{train_ratio}.txt', train_scans[train_ratio])
    save_split('valid.txt', valid_scans)
    save_split('test.txt', test_scans)

    logs = [f'split\t\tn_scans\tpos\tneg\tn_days\tpos\tneg\n']
    def add_log(split, days, scans):
        n_scans = len(scans)
        n_pos_scans = len(set(scans).intersection(station_years[station_year]["scans_with_roosts"]))
        n_neg_scans = n_scans - n_pos_scans
        logs.append(
            f"{split}\t{n_scans}\t{n_pos_scans}\t{n_neg_scans}\t"
            f"{len(days)}\t"
            f"{len(set(days).intersection(station_years[station_year]['roost_days']))}\t"
            f"{len(set(days).intersection(station_years[station_year]['non_roost_days']))}\n"
        )

    for train_ratio in TRAIN_RATIOS:
        add_log(f"train_{train_ratio}", train_days[train_ratio], train_scans[train_ratio])
    add_log("valid\t", valid_days, valid_scans)
    add_log("test\t", test_days, test_scans)
    with open(os.path.join(splits_dir, 'stats.txt'), "w") as f:
        f.writelines(logs)


""" dbz ordered sampling
    dbz = [np.mean([
        station_years[station_year]['all_scans_with_check'][scan]['avg_dbz']
        for scan in station_years[station_year]['all_days_to_scans'][day]
    ]) for day in non_test_days]
    non_test_days = [day for day, _ in sorted(zip(non_test_days, dbz), key=lambda p: -p[1])]
    n_non_test_days = len(non_test_days)

    top_dbz_days = non_test_days[:int(n_non_test_days / 4)]  # top quartile
    random.shuffle(top_dbz_days)
    other_dbz_days = non_test_days[int(n_non_test_days / 4):]
    random.shuffle(other_dbz_days)

    n_train_days = int(TRAIN_RATIOS[-1] * n_days)
    n_val_days = int(VALID_RATIO * n_days)
    if (n_train_days + n_val_days) / 2 > n_non_test_days / 4:
        # get the entire top quartile
        train_days[TRAIN_RATIOS[-1]].extend(top_dbz_days[:int(n_train_days / (n_train_days + n_val_days))])
        val_days.extend(top_dbz_days[int(n_train_days / (n_train_days + n_val_days)):])
        # sample from the rest
        other_dbz_days = other_dbz_days[:n_train_days + n_val_days - len(top_dbz_days)]
        train_days.extend(other_dbz_days[:int(n_train_days / (n_train_days + n_val_days))])
        val_days.extend(other_dbz_days[int(n_train_days / (n_train_days + n_val_days)):])
    else:
        # half from top quartile
        top_dbz_days = top_dbz_days[:int((n_train_days + n_val_days) / 2)]
        train_days.extend(top_dbz_days[:int(len(top_dbz_days) * n_train_days / (n_train_days + n_val_days))])
        val_days.extend(top_dbz_days[int(len(top_dbz_days) * n_train_days / (n_train_days + n_val_days)):])
        # half from the rest
        other_dbz_days = other_dbz_days[:int((n_train_days + n_val_days) / 2)]
        train_days.extend(other_dbz_days[:int(len(other_dbz_days) * n_train_days / (n_train_days + n_val_days))])
        val_days.extend(other_dbz_days[int(len(other_dbz_days) * n_train_days / (n_train_days + n_val_days)):])
"""