import boto3
from datetime import datetime, timedelta
import errno
import os
import re
import sys


####################################
# Helpers
####################################
def datetime_range(start=None, end=None, delta=timedelta(minutes=1), inclusive=False):
    """Construct a generator for a range of dates

    Args:
        from_date (datetime): start time
        to_date (datetime): end time
        delta (timedelta): time increment
        inclusive (bool): whether to include the 

    Returns:
        Generator object
    """
    t = start or datetime.now()

    if inclusive:
        keep_going = lambda s, e: s <= e
    else:
        keep_going = lambda s, e: s < e

    while keep_going(t, end):
        yield t
        t = t + delta
    return


def s3_key(t, station):
    """Construct (prefix of) s3 key for NEXRAD file

    Args:
        t (datetime): timestamp of file
        station (string): station identifier

    Returns:
        string: s3 key, excluding version string suffic

    Example format:
            s3 key: 2015/05/02/KMPX/KMPX20150502_021525_V06.gz
        return val: 2015/05/02/KMPX/KMPX20150502_021525
    """

    key = '%04d/%02d/%02d/%04s/%04s%04d%02d%02d_%02d%02d%02d' % (
        t.year,
        t.month,
        t.day,
        station,
        station,
        t.year,
        t.month,
        t.day,
        t.hour,
        t.minute,
        t.second
    )

    return key


def s3_prefix(t, station=None):
    prefix = '%04d/%02d/%02d' % (
        t.year,
        t.month,
        t.day
    )

    if not station is None:
        prefix = prefix + '/%04s/%04s' % (station, station)

    return prefix


def parse_key(key):
    path, key = os.path.split(key)
    vals = re.match('(\w{4})(\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2})(\.?\w+)', key)
    (station, timestamp, suffix) = vals.groups()
    t = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
    return t, station


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


####################################
# AWS setup
####################################
# start_time = datetime(2015, 5, 2,  0,  0,  0)
# end_time   = datetime(2015, 5, 2,  0, 29, 59)
# stations = ['KLIX', 'KLCH', 'KLIX']
stride_in_minutes = 3
thresh_in_minutes = 3
bucket = boto3.resource('s3', region_name='us-east-2').Bucket('noaa-nexrad-level2')
darkecology_bucket = boto3.resource('s3', region_name='us-east-2').Bucket('cajun-batch-test')


def get_scans(start_time, end_time, stations, select_by_time=False, time_increment=None, stride_increment=None,
              thresh_increment=None, with_station=True):
    #################
    # First get a list of all keys that are within the desired time period
    # and divide by station 
    #################
    all_keys = []
    keys_by_station = {s: [] for s in stations}

    if not time_increment:
        time_increment = timedelta(days=1)

    if not stride_increment:
        stride_increment = timedelta(minutes=stride_in_minutes)

    if not thresh_increment:
        thresh_increment = timedelta(minutes=thresh_in_minutes)

    for station in stations:
        for t in datetime_range(start_time, end_time, time_increment, inclusive=True):
            # Set filter
            prefix = s3_prefix(t, station)
            # print prefix

            start_key = s3_key(start_time, station)
            end_key = s3_key(end_time, station)

            # Get s3 objects for this day
            objects = bucket.objects.filter(Prefix=prefix)

            # Select keys that fall between our start and end time
            keys = [o.key for o in objects
                    if o.key >= start_key
                    and o.key <= end_key]

            # Add to running lists
            all_keys.extend(keys)
            keys_by_station[station].extend(keys)
    # print(all_keys)

    #################
    # Now iterate by time and select the appropriate scan for each station
    #################
    time_thresh = thresh_increment  # timedelta( minutes = thresh_in_minutes )
    times = list(datetime_range(start_time, end_time, stride_increment))
    current = {s: 0 for s in stations}
    selected_by_time = {t: set() for t in times}
    # selected_by_station = { s: set() for s in stations }
    selected_keys = []

    for t in times:
        for station in stations:
            keys = keys_by_station[station]
            i = current[station]

            if keys:
                t_current, _ = parse_key(keys[i])

                while i + 1 < len(keys):
                    t_next, _ = parse_key(keys[i + 1])
                    if abs(t_current - t) < abs(t_next - t):
                        break
                    t_current = t_next
                    i = i + 1

                current[station] = i
                k = keys[i]

                if abs(t_current - t) <= time_thresh:
                    if select_by_time:
                        selected_by_time[t].add(k)
                    # selected_by_station[station].add(k)
                    if with_station:
                        selected_keys.append("%s;%s" % (k, station))
                    else:
                        selected_keys.append(k)

    return selected_keys if not select_by_time else selected_by_time


def download_scans(keys, data_dir):
    #################
    # Download files into hierarchy
    #################
    for key in keys:
        # Download files
        local_file = os.path.join(data_dir, key)
        local_path, filename = os.path.split(local_file)
        mkdir_p(local_path)

        # Download file if we don't already have it
        if not os.path.isfile(local_file):
            bucket.download_file(key, local_file)
