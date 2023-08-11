from wsrdata.utils.s3_utils import download_scans
from botocore.exceptions import ClientError
import logging
import time


# inputs a txt file where each line is a scan name, e.g.
# KOKX20130721_093320_V06
# KTBW20031123_115217
def download_by_scan_list(filepath, out_dir, log_path,
                          not_s3_log_path, # scans that are not found in s3
                          error_scans_log_path):

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        filelog = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s [ %(fname)s ] : %(message)s')
        formatter.converter = time.gmtime
        filelog.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(filelog)
    logger = logging.LoggerAdapter(logger, {"fname": filepath})

    logger.info('***** Start downloading for %s *****' % (filepath))

    # Load all scans
    scans = [scan.strip() for scan in open(filepath, "r").readlines()] # Load all scans
    not_s3 = [] # record scans not in s3
    error_scans = [] # record scans whose downloading fails due to reasons other than not in s3

    # download each scan
    for scan in scans:
        try:
            scan = '%s.gz' % scan
            station = scan[0:4]
            year = scan[4:8]
            month = scan[8:10]
            date = scan[10:12]
            aws_key = '%s/%s/%s/%s/%s' % (year, month, date, station, scan)
            print(aws_key)
            download_scans([aws_key], out_dir)
            logger.info('Downloaded scan %s, aws key %s' % (scan, aws_key))
        except ClientError as err:
            error_code = int(err.response['Error']['Code'])
            if error_code == 404:
                logger.error('Error Scan %s not found in s3, adding to list' % scan)
                not_s3.append(scan)
            else:
                logger.error('Exception while processing scan %s - %s' % (scan, str(err)))
                error_scans.append(scan)
        except Exception as ex:
            logger.error('Exception while processing scan %s - %s' % (scan, str(ex)))
            error_scans.append(scan)

    if len(not_s3) > 0:
        with open(not_s3_log_path, 'a+') as f:
            f.write('\n'.join(not_s3)+'\n')

    if len(error_scans) > 0:
        with open(error_scans_log_path, 'a+') as f:
            f.write('\n'.join(error_scans)+'\n')

    logger.info('***** Finished downloading for file %s *****' % (filepath))
    return {"not_s3": not_s3, "error_scans": error_scans}