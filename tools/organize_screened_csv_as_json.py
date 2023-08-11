"""
This script is a modifiable template which is written for dataset v0.2.B,
where B indicates different sampling strategies.
It organizes scans and annotations from multiple years of a station and save them as json,
and calculates stats of ecologist-screened roost-system predictions.
This script process one station at a run because
loading npz files to calculate average dbz and checking dualpol can be slow.
"""

import argparse
import json
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--station", type=str, required=True, help="station name")
args = parser.parse_args()

SYS_PRED_DIR = '/scratch2/wenlongzhao/roostui/data/all_stations_v2'
SYS_START_DATE = '0601'
SYS_END_DATE = '1231'
    # files import to the UI, we use scan lists from here
    # eg. scans_KAPX_20200601_20201231.txt
    #     tracks_KAPX_20200601_20201231.txt
ARRAY_NPZ_DIR = '/scratch2/wenlongzhao/RadarNPZ/v0.2.0'
    # npz files to store rendered arrays
    # eg. 2020/06/01/KAPX/KAPX20200601_092853_V06.npz
SCREENED_ANNOTATION_CSV_DIR = '../static/annotations/v2.0.0/csv'
    # csv files output from the UI
    # eg. roost_labels_KAPX_20200601_20201231.csv
MONTHS = [6, 10] # system predictions from June to Oct are screened, left and right inclusive

# Figure out which station-years we are interested in
station_years = {} # key: station_year, value: a dictionary for this station-year
for file in sorted(os.listdir(SCREENED_ANNOTATION_CSV_DIR)):
    if args.station not in file: continue
    station_years['_'.join((file.split('_')[2], file.split('_')[3][:4]))] = {
        'all_scans_with_check':     {},  # scan: {"avg_dbz": float, "dualpol": True/False}
        'all_days_to_scans':        {},  # day: set(scan)

        'n_roost_annotations':              0,
        'n_roost_annotations_not_miss_day': 0,
        'n_bad_track_annotations':          0,
        'scans_with_roosts':                set(),  # positive scans
        'roost_days':                       set(),  # days with roosts

        'n_scans_without_roosts_in_roost_days': 0,  # negative scans
        'n_scans_in_non_roost_days':            0,  # scans from sampled non_roost_days become negatives
        'non_roost_days':                       set(),  # days without roosts
    }
print(f'There are {len(station_years)} years for station {args.station} from the csv files that we\'re interested in.')
print(f'Sample station-years: {list(station_years.keys())[:5]}.\n')

# For each station-year of interest, read the corresponding scan list
for station_year in station_years:
    station, year = station_year.split("_")
    # Read the list of scans with successfully rendered arrays for relevant months of this station-year.
    # For each scan, calc the average dbz at the lowest elevation and check if a dualpol array exists.
    # Fill all_scans_with_check and all_days_to_scans.
    sys_pred_scan_list = f'scans_{station}_{year}{SYS_START_DATE}_{year}{SYS_END_DATE}.txt'
    for scan in open(os.path.join(SYS_PRED_DIR, sys_pred_scan_list), 'r').readlines()[1:]:
        scan = scan.strip().split(",")[0]
        if int(scan[8:10]) < MONTHS[0] or int(scan[8:10]) > MONTHS[1]:
            continue
        npz_path = os.path.join(ARRAY_NPZ_DIR, f'{scan[4:8]}/{scan[8:10]}/{scan[10:12]}/{scan[:4]}/{scan}.npz')
        assert os.path.exists(npz_path), f'{scan} does not have an npz'
        arrays = np.load(npz_path)
        station_years[station_year]['all_scans_with_check'][scan] = {
            'avg_dbz':  np.mean(np.nan_to_num(arrays['array'][0, 0, :, :], copy=False, nan=0.0)),
            'dualpol':  'dualpol_array' in arrays,
        }
        if scan[4:12] not in station_years[station_year]['all_days_to_scans']:
            station_years[station_year]['all_days_to_scans'][scan[4:12]] = set()
        assert scan not in station_years[station_year]['all_days_to_scans'][scan[4:12]]
        station_years[station_year]['all_days_to_scans'][scan[4:12]].add(scan)

    # Read annotations for relevant months of this station-year. Check that they're are viewed.
    screened_annotation_csv = f'roost_labels_{station}_{year}{SYS_START_DATE}_{year}{SYS_END_DATE}.csv'
    csv_file = os.path.join(SCREENED_ANNOTATION_CSV_DIR, screened_annotation_csv)
    annotations = [annotation.strip().split(",") for annotation in open(csv_file, "r").readlines()[1:]]
    # fields are:
    #       0 track_id, 1 filename, 2 from_sunrise, 3 det_score, 4 x, 5 y, 6 r, 7 lon, 8 lat, 9 radius,
    #       10 local_time, 11 station, 12 date, 13 time, 14 local_date, 15 length,
    #       16 tot_score, 17 avg_score, 18 viewed, 19 user_labeled, 20 label, 21 original_label,
    #       22 notes: 'LARGE', 'nr', 'long', 'large', 'rn', 'shrinks', 'shrink',
    #       23 day_notes: 'pap', 'psp', 'weather', '2', 'ap', 'AP', 'miss', 'cluster', 'clusters',
    for annotation in annotations:
        # skip if not in the months of interest
        if int(annotation[1][8:10]) < MONTHS[0] or int(annotation[1][8:10]) > MONTHS[1]:
            continue
        # if not viewed, i.e. not screened by ecologists, consider as non-roost and skip
        if annotation[18].lower() != 'true':
            # print("Unexpected not viewed:", annotation[1])
            continue
        # verify that the scan and day are in the scan list
        assert annotation[1] in station_years[station_year]['all_scans_with_check']
        assert annotation[1][4:12] in station_years[station_year]['all_days_to_scans']
        # process the annotation based on the label
        if annotation[20] in ['non-roost', 'duplicate']:
            continue
        elif annotation[20] == 'bad-track':
            station_years[station_year]['n_bad_track_annotations'] += 1
        else:
            assert 'roost' in annotation[20]
            station_years[station_year]['n_roost_annotations'] += 1
            if annotation[23].lower() != 'miss':
                station_years[station_year]['n_roost_annotations_not_miss_day'] += 1
            station_years[station_year]['scans_with_roosts'].add(annotation[1])
            assert annotation[1][4:12] == annotation[12]
            station_years['_'.join((station, year))]['roost_days'].add(annotation[12])

    # Collect scans and days without roosts
    for day in station_years[station_year]['all_days_to_scans']:
        if day in station_years[station_year]['roost_days']:
            for scan in station_years[station_year]['all_days_to_scans'][day]:
                if scan not in station_years[station_year]['scans_with_roosts']:
                    station_years[station_year]['n_scans_without_roosts_in_roost_days'] += 1
        else:
            station_years[station_year]['non_roost_days'].add(day)
            for scan in station_years[station_year]['all_days_to_scans'][day]:
                station_years[station_year]['n_scans_in_non_roost_days'] += 1

    # turn sets into lists so they are serializable
    for day in station_years[station_year]['all_days_to_scans']:
        station_years[station_year]['all_days_to_scans'][day] = sorted(list(
            station_years[station_year]['all_days_to_scans'][day]
        ))
    station_years[station_year]['scans_with_roosts'] = sorted(list(station_years[station_year]['scans_with_roosts']))
    station_years[station_year]['roost_days'] = sorted(list(station_years[station_year]['roost_days']))
    station_years[station_year]['non_roost_days'] = sorted(list(station_years[station_year]['non_roost_days']))

    print(station_year, 'done.')

with open(f'prepare_dataset_v0.2.0_help/all_days_all_scans_{args.station}.json', 'w') as f:
    json.dump(station_years, f)