from wsrlib import pyart, radar2mat
import logging
import time
import os
import numpy as np


# inputs a txt file where each line is a scan name, e.g.
# KOKX20130721_093320_V06
# KTBW20031123_115217
def render_by_scan_list(filepath, scan_dir, array_dir,
                        array_render_config, dualpol_render_config,
                        force_rendering=False):

    log_path = os.path.join(array_dir, "rendering.log")
        # this includes successful rendering for arrays and dualpol arrays
    array_error_log_path = os.path.join(array_dir, "array_error_scans.log")
    dualpol_error_log_path = os.path.join(array_dir, "dualpol_error_scans.log")

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        filelog = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s [ %(fname)s ] : %(message)s')
        formatter.converter = time.gmtime
        filelog.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(filelog)
    logger = logging.LoggerAdapter(logger, {"fname": filepath})

    logger.info('***** Start rendering for %s *****' % (filepath))

    scans = [scan.strip() for scan in open(filepath, "r").readlines()] # Load all scans
    array_errors = [] # to record scans from which array rendering fails
    dualpol_errors = [] # to record scans from which dualpol array rendering fails

    # render arrays from scans
    for scan in scans:
        station = scan[0:4]
        year = scan[4:8]
        month = scan[8:10]
        date = scan[10:12]
        scan_file = os.path.join(scan_dir, f"{year}/{month}/{date}/{station}/{scan}.gz")
        arrays = {}
        npz_path = os.path.join(array_dir, f"{year}/{month}/{date}/{station}/{scan}.npz")

        if os.path.exists(npz_path):
            if force_rendering:
                arrays = np.load(npz_path)
            else:
                logger.info('Rendered arrays already exist for scan %s' % scan)
                continue

        try:
            radar = pyart.io.read_nexrad_archive(scan_file)
            logger.info('Loaded scan %s' % scan)
        except Exception as ex:
            logger.error('Exception while loading scan %s - %s' % (scan, str(ex)))
            array_errors.append(scan)
            dualpol_errors.append(scan)
            continue

        try:
            data, _, _, y, x = radar2mat(radar, **array_render_config)
            logger.info('Rendered a npy array from scan %s' % scan)
            if data.shape != (len(array_render_config["fields"]), len(array_render_config["elevs"]),
                              array_render_config["dim"], array_render_config["dim"]):
                logger.info(f"  Unexpectedly, its shape is {data.shape}.")
            arrays["array"] = data
        except Exception as ex:
            logger.error('Exception while rendering a npy array from scan %s - %s' % (scan, str(ex)))
            array_errors.append(scan)

        try:
            data, _, _, y, x = radar2mat(radar, **dualpol_render_config)
            logger.info('Rendered a dualpol npy array from scan %s' % scan)
            if data.shape != (len(dualpol_render_config["fields"]), len(dualpol_render_config["elevs"]),
                              dualpol_render_config["dim"], dualpol_render_config["dim"]):
                logger.info(f"  Unexpectedly, its shape is {data.shape}.")
            arrays["dualpol_array"] = data
        except Exception as ex:
            logger.error('Exception while rendering a dualpol npy array from scan %s - %s' % (scan, str(ex)))
            dualpol_errors.append(scan)

        if len(arrays) > 0:
            os.makedirs(os.path.join(array_dir, f"{year}/{month}/{date}/{station}"), exist_ok=True)
            np.savez_compressed(npz_path, **arrays)

    if len(array_errors) > 0:
        with open(array_error_log_path, 'a+') as f:
            f.write('\n'.join(array_errors)+'\n')
    if len(dualpol_errors) > 0:
        with open(dualpol_error_log_path, 'a+') as f:
            f.write('\n'.join(dualpol_errors)+'\n')

    logger.info('***** Finished rendering for file %s *****' % (filepath))
    return array_errors, dualpol_errors