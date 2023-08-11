This directory contains helper functions that can be useful for accelerating the creation of 
dataset v0.1.0 and checking the correctness of the creation.

Since the dataset is pretty large, we split `static/scan_lists/v0.1.0/v0.1.0_ordered_splits/{train,test}.txt` 
into `train_*.txt` and `test_*.txt` under this directory.
`static/scan_lists/v0.1.0/v0.1.0_ordered_splits/val.txt` is not split since it's comparatively small.

Run `tools/prepare_dataset_v0.1.0_dl_rd.py` multiple times for parallel downloading and rendering for the data subsets.

For record, we copy the generated
`static/arrays/v0.1.0/{rendering.log, array_error_scans.log, dualpol_error_scans.log}` to this directory.

Typically `previous_versions.json` under `static/arrays` will be updated to avoid future version conflicts.
As it is not done by `prepare_dataset_v0.1.0_dl_rd.py`, use `log_array_version.py` for a manual update.

Run `handle_exceptions.py` to (1) print scans that are not downloaded, (2) collect unexpected errors from 
`rendering.log` and write to `rendering_exceptions.txt`, and (3) create 
`static/scan_lists/v0.1.0/v0.1.0_standard_splits` which contain scans without exceptions.

Run `tools/prepare_dataset_v0.1.0.py` which is configured to skip downloading and rendering and 
directly create json files.

Run `tools/prepare_dataset_v0.1.0_raw.py` for not scaling/standardizing the bounding boxes.