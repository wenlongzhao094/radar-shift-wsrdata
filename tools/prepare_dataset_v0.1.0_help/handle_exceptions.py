import os


# make sure all scans are successfully downloaded
print("Start printing scans that are not downloaded.")
scan_dir = "../../static/scans/scans"
filepath = "../../static/scan_lists/v0.1.0/scan_list.txt"
scans = [scan.strip() for scan in open(filepath, "r").readlines()]
for scan in scans:
    station = scan[0:4]
    year = scan[4:8]
    month = scan[8:10]
    date = scan[10:12]
    scan_file = os.path.join(scan_dir, '%s/%s/%s/%s/%s.gz' % (year, month, date, station, scan))
    assert os.path.exists(scan_file)
print("End printing scans that are not downloaded.")


# rendering exceptions from rendering.log
print("Saving exceptions to rendering_exceptions.txt...")
filepath = "rendering.log" # "../static/arrays/v0.1.0/rendering.log"
logs = [log.strip().split(" : ")[1] for log in open(filepath, "r").readlines()]

exceptions = []
scans_with_exceptions = []
for log in logs:
    if log.startswith("Exception") and \
            not log.startswith("Exception while rendering a dualpol npy array from scan"):
        exceptions.append(log+"\n")
        scans_with_exceptions.append(log.split(" scan ")[1].split(" - ")[0])
with open("rendering_exceptions.txt", "w") as f:
    f.writelines(exceptions)
print("Done.")

# Removing scans with exceptions from scan_lists/v0.1.0/v0.1.0_ordered_splits and
# create v0.1.0_standard
scans_with_exceptions = set(scans_with_exceptions)
assert scans_with_exceptions == set([scan.strip() for scan in open("array_error_scans.log", "r").readlines()])
os.makedirs(f"../../static/scan_lists/v0.1.0/v0.1.0_standard_splits", exist_ok=True)
for split in ["train", "val", "test"]:
    with open(f"../../static/scan_lists/v0.1.0/v0.1.0_ordered_splits/{split}.txt", "r") as f:
        scans = f.readlines()
    with open(f"../../static/scan_lists/v0.1.0/v0.1.0_standard_splits/{split}.txt", "w") as f:
        f.writelines([scan for scan in scans if scan.strip() not in scans_with_exceptions])

