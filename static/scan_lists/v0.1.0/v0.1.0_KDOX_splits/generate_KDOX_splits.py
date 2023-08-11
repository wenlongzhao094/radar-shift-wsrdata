for split in ["train.txt", "val.txt", "test.txt"]:
    scans = open(f"../v0.1.0_standard_splits/{split}", "r").readlines()
    with open(split, "w") as f: f.writelines([scan for scan in scans if scan.startswith("KDOX")])