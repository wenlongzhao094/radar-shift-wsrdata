"""
Pick 30 random scans with annotations for each annotater-station pair
Make scan list named annotator-STATION.txt with these scans
Later use tools/visualize.py to generate png images to visualize these scans with bounding boxes
See if the boxes make sense
NOTE that sometimes multiple annotators may annotate a scan -- a scan in annotator-STATION.txt does not mean
all annotations for the scan is by that annotator.
"""
import random
random.seed(2021)

annotator_station_pairs = ['Ftian-KOKX', 'William Curran-KDOX', 'andrew-KAMX', 'andrew-KHGX', 'andrew-KJAX',
                           'andrew-KLCH', 'andrew-KLIX', 'andrew-KMLB', 'andrew-KTBW', 'andrew-KTLH',
                           'anon-KDOX', 'anon-KLIX', 'anon-KTBW', 'jafer1-KDOX', 'jafermjj-KDOX',
                           'jberger1-KAMX', 'jberger1-KLIX', 'jberger1-KMLB', 'jberger1-KTBW', 'jpodrat-KLIX',
                           'sheldon-KAMX', 'sheldon-KDOX', 'sheldon-KLIX', 'sheldon-KMLB', 'sheldon-KOKX',
                           'sheldon-KRTX', 'sheldon-KTBW']
scans_with_annotations = {pair: [] for pair in annotator_station_pairs}

scans = [scan.strip() for scan in open("../v0.1.0_standard_splits/train.txt", "r").readlines()]
scans.extend([scan.strip() for scan in open("../v0.1.0_standard_splits/val.txt", "r").readlines()])
scans.extend([scan.strip() for scan in open("../v0.1.0_standard_splits/test.txt", "r").readlines()])
scans = set(scans)

error_scans = set([scan.strip() for scan in open(
    "/scratch2/wenlongzhao/roosts2021/libs/wsrdata/tools/prepare_dataset_v0.1.0_help/array_error_scans.log",
    "r").readlines()])

# annotations ordered alphabetically by scan, then by date, then by track, but not by second
# fields are: 0 scan_id (different than ours), 1 filename, 2 sequence_id, 3 station, 4 year, 5 month,
#         6 day, 7 hour, 8 minute, 9 second, 10 minutes_from_sunrise, 11 x, 12 y, 13 r, 14 username
annotations = [annotation.strip().split(",") for annotation in
               open("../../../annotations/v1.0.0/user_annotations.txt", "r").readlines()[1:]]
for annotation in annotations:
    assert annotation[1].split(".")[1] in ["gz", "Z"]
    scan = annotation[1].split(".")[0]
    if scan in scans and scan not in error_scans \
            and annotation[-1] in scans_with_annotations \
            and scan not in scans_with_annotations[annotation[-1]]:
        scans_with_annotations[annotation[-1]].append(scan)

for pair in scans_with_annotations:
    output_scan_list = []
    n_candidates = len(scans_with_annotations[pair])
    if n_candidates < 30:
        print(f"Warning: {pair} has {n_candidates} scans in dataset v0.1.0.")
    else:
        print(f"{pair} has {n_candidates} scans in dataset v0.1.0.")
    indices = sorted(random.sample(range(n_candidates), min(30, n_candidates)))
    for i in indices:
        output_scan_list.append(scans_with_annotations[pair][i]+"\n")
    if len(output_scan_list) != 0:
        open(pair+".txt", "w").writelines(output_scan_list)
