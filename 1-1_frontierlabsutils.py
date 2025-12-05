"""
frontierlabsutils.py

by Tessa Rhinehart
2023-10-16

Utilities for using Frontier Labs BAR-LT
synchronized autonomous acoustic recorders

To adapt for your own recordings you need to change: data directory, recorder list,
path wildcards in some functions (get_recording_path, get_overflows, get_loclog_contents),
and how to get recordings from the same recording period (get_all_times)

"""

import pytz
import datetime

from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram

from pathlib import Path
from glob import glob
from itertools import combinations
from math import sqrt, floor
from scipy.stats import mode
from scipy.signal import resample

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import warnings

print("Importing Frontier Labs Utilities")
#set directory where your recordings are stored
DATA_DIR = "C:/Users/AlexE/OneDrive - EC-EC/Robinson,Barry (il _ he, him) (ECCC)'s files - Grassland Bird Monitoring/R Projects/Localization/recordings/snas_cclo_follow"

### RECORDING NAME UTILITIES ###

def get_recorder_list():
    """Returns list of 49 recorder names A1-G7.
    """
    cols = ["A","B","C","D","E","F","G"]
    rows = ["1","2","3","4","5","6","7"]
    recorders = [c+r for c in cols for r in rows]
    
    return recorders

class InputError(Exception):
    """Custom exception for input errors."""
    pass

def create_localized_datetime(dt_string):
    """Convert a string to a timezone-localized datetime obj
    
    Args:
        dt_string: string of datetime info, of format:
            20220408T155959.129679+1000
            YYYYMMDD_HHMMSS.MICROS+ZONE
            years, months, days, hours, minutes, seconds, microseconds, +/- timezone
    """
        
    def _split_time_and_zone(time_and_zone):  
        if '+' in time_and_zone:
            time, zone = time_and_zone.split('+')
            zone = '+' + zone
        elif '-' in time_and_zone:
            time, zone_inverted = time_and_zone.split('-')
            zone = '-' + zone_inverted
        else:
            raise InputError("Input time zone should have either a + or a -")
        return time, zone
    
    def _get_year_month_day(date):
        return int(date[:4]), int(date[4:6]), int(date[6:])
    
    def _get_hh_mm_ss_microsec(time):
        return int(time[:2]), int(time[2:4]), int(time[4:6]), int(time[7:])
    
    def _format_tz(zone):
        if '+' in zone:
            multiplier = 1
        elif '-' in zone:
            multiplier = -1
        else:
            raise InputError("Input time zone should have either a + or a -")
            
        delta = datetime.timedelta(
                hours=multiplier*int(zone[1:3]),
                minutes=int(zone[3:]))
        return datetime.timezone(delta)
    
    
    date, time_and_zone = dt_string.split('T')
    
    ymd = _get_year_month_day(date)
    
    time, zone = _split_time_and_zone(time_and_zone)
    hms_ms = _get_hh_mm_ss_microsec(time)
    tz = _format_tz(zone)
    
    d = datetime.datetime(
        year=ymd[0],
        month=ymd[1],
        day=ymd[2],
        hour=hms_ms[0],
        minute=hms_ms[1],
        second=hms_ms[2],
        microsecond=hms_ms[3],
        tzinfo=tz
    )
    
    return d

def extract_start_end(filename):
    """Extract start and end time of filename from Frontier Labs recorder
    
    Inputs:
    - filename: the filename (not the path) of the file
    """
  
    split_up = filename.split('_')
    start = split_up[0]
    end = split_up[1]
    GPS = split_up[2]
    start_datetime = create_localized_datetime(start.strip('S'))
    end_datetime = create_localized_datetime(end.strip('E'))
    
    return start_datetime, end_datetime

def get_rec_name(*args, **kwargs):
    """Alias for get_recording_path()
    """
    return get_recording_path(args, kwargs)

def get_recording_path(
    recorder,
    date,
    hour_minute,
    data_dir,
    valid_times=None,
    logfile_name=None,
    logging=False
):
    """Get path of recording given a recorder, time, and date
    
    This function tries to get the path to a recording from 
    a particular recorder and time. If one cannot be found,
    it prints this. Additionally, these results can be logged to a logfile.
    
    Arguments:
        recorder (string): string with one capital letter 
            (A-G) and one number (1-7) e.g. A1
        date (string): day formatted YYYYMMDD e.g. 20230628
        hour_minute (string): time formatted HHMM e.g. 0459
            Available times are between 03:59 to 09:59;
            mins are either 29 or 59
            
            if hour_minute == "any", this function will return 
            all recordings from this date
        valid_times (list): list that hour_minute should be in
            or, if valid_times=="playback", will check that
            hour_minute is in one of the playback times
        data_dir: top-level directory to search for recording 
            in. Change this to resampled directory if needed.
        logfile_name (string path): path of logfile to optionally 
            save information to about files that aren't found.
        logfile (bool): whether or not to log results to a file
    
    Returns:
        Return format depends on hour_minute:
            if hour_minute != "any": returns the recording from 
                this date, or None if no files
            if hour_minute == "any": a list of all recordings from 
                this date, or an empty array if no files
    """
    if "resampled" in data_dir:
        if hour_minute == "any":
            recordings = list(Path(data_dir, recorder).glob(
                f"{recorder}_S{date}*.wav"))
        else:
            recordings = list(Path(data_dir, recorder).glob(f"{recorder}_S{date}T{hour_minute}*.wav"))
    else:
        if hour_minute == "any":
            recordings = list(Path(data_dir, recorder).glob(
                f"S{date}*.wav"))
        else:
            recordings = list(Path(data_dir, recorder).glob(
                f"S{date}T{hour_minute}*.wav"))
    if valid_times == "playback":
        valid_times = ["0359", "0429", "0459", 
                       "0529", "0559", "0629", 
                       "0659", "0729", "0759", 
                       "0829", "0859", "0929",
                       "0959"]
    if valid_times is not None:
        if hour_minute not in valid_times:
            raise ValueError("hour_minute must be one of: "+str(valid_times))
    
    if len(recordings)<1:
        print(f"recorder {recorder} - day {date} - time {hour_minute} not found")
        if logging:
            f = open(logfile_name, "a+")
            f.write(f"recorder {recorder} - day {date} - time {hour_minute} not found\n")
            f.close()
        if hour_minute=="any":
            return None
        else:
            return []
    elif len(recordings)==1:
        if hour_minute=="any":
            return recordings
        else:
            return recordings[0]
    elif hour_minute != "any":
        raise ValueError(f"multiple recordings for {date} and {hour_minute}")
    else:
        return recordings
    
def get_all_times(time):
    """Get all possible recording minutes given a potential start time
    
    Arguments:
        time: hour and minute string in the format '0429', '0958', etc.
            minute must be either 28, 29, 30 or 58, 59, 00
            
    Returns:
        list of all possible recording minute strings, helpful for
        use with finding overflows
        
    Example:
        get_all_times("0458") --> ['0458', '0459', '0500', '0501', 
            '0502', '0503', '0504', '0505', '0506', '0507', '0508', 
            '0509', '0510']
        
    """
    hour = time[:2]
    minute = time[2:]
    
    time_list = []
    if int(minute) > 31:
        hour1 = hour
        hour2 = int(hour)+1
        minutes_hour1 = range(int(minute), 60)
        minutes_hour2 = range(0, 11)
        for minute in minutes_hour1:
            time_list.append(str(hour1).zfill(2)+str(minute).zfill(2))
        for minute in minutes_hour2:
            time_list.append(str(hour2).zfill(2)+str(minute).zfill(2))
    
    else:
        minutes_hour = range(int(minute), 41)
        for minute in minutes_hour:
            time_list.append(str(hour).zfill(2) + str(minute).zfill(2))
    return(time_list)
    print(minute)

    
    
### SYNCHRONIZATION UTILITIES ###

def get_overflows(
    recorder, data_dir=DATA_DIR, logfile_name=None, logging=False):
    """Get all numbers of overflows from txt files for a recorder
    
    Finds all logfiles for a recorder and gets the dates of the overflows.
    
    Arguments:
        recorder (string): string with one capital letter (A-G) and one number (1-7) e.g. A1
        
    Returns:
        dataframe containing the number of overflows for all recordings for all recorders
    """
    txtfiles = list(glob(f"{data_dir}/{recorder}*/*txt"))
    
    if len(txtfiles)<1:
        info_string = f"recorder {recorder} - txt file not found"
        print(info_string)
        if logging:
            f = open(logfile_name, "a+")
            f.write(info_string+'\n')
            f.close()
        return None

    overruns = []
    for txtfile in txtfiles:
        df = pd.read_csv(txtfile, sep='\t', encoding="unicode_escape")
        for c in df.values:
            if "overruns" in c[0]:
                
                # Extract info from the line
                date, time, t, b, o, e, num_overruns = c[0].split(' ')
                hour = int(time[:2])
                minute = int(time[3:5])
                
                # Format the recording date and start time
                start_time_1900s = datetime.datetime(
                    year=1900,
                    month=1,
                    day=1,
                    hour=hour,
                    minute=minute) - datetime.timedelta(minutes=11)
                date_string = ''.join(date.split('-'))
                start_string = start_time_1900s.strftime("%H%M")
                
                # Add to list of overruns
                overruns.append([recorder, date_string, start_string, int(num_overruns)])

    return pd.DataFrame(overruns, columns=["recorder", "date_str", "time_str", "num_overruns"])

def format_write_time(write_time_str, date_format):
    """Format string write time to datetime.timedelta
    
    Returns: datetime.timedelta object containing write time 
    """
    time = write_time_str[len(date_format)+1:].split(" -> ")[0][:-len("-6:00")-1]
    try:
        h, m, sms = time.split(':')
    except:
        split_output = time.split(":")
        raise ValueError(
            "write_time_str did not split correctly; got", 
            write_time_str, "which produced output", split_output)
    t = datetime.timedelta(hours=int(h), minutes=int(m), seconds=float(sms))
    return t

#a = format_write_time('2023-06-16T03:59:00.039561-06:00 -> 740203140.039561', "2023-06-18")
#assert a == datetime.timedelta(seconds=14340, microseconds=39561)

def get_recording_write_times(logfile_values, date, start_time):
    """Get write times for a recording starting at a particular write time
    
    Arguments:
        logfile_times: list of lines from logfile
        date: string formatted date e.g. "20230628"
        time: the start_time of the recording
    
    Returns:
        list of timestamps for write_times
    """
    write_times = []
    for val in logfile_values:
        date_format = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        # Get all valid start times for this particular recording
        try:
            valid_times = get_all_times(start_time)
        except:
            raise ValueError("Start time invalid. Start time given:", start_time)
            print(start_time)
        
        # Get all lines from this start_time
        valid_timestamps = [date_format + f"T{time[:2]}:{time[2:]}" for time in valid_times]
        for stamp in valid_timestamps:
            if stamp in val[0]:
                write_times.append(format_write_time(val[0], date_format)) 
    return write_times

def get_loclog_contents(data_dir, recorder, date):
    """Find logfile(s) and all contents
    
    Arguments:
        data_dir (string): top-level dir such that this format pattern leads to the logfile:
            f"{data_dir}/MIN231x{recorder}*/*{date}*/loclog.txt"
        recorder (string): recorder string e.g. "A1", "G7"
        date (string): date string formatted YYYYMMDD e.g. "20230628"
    
    Returns: 
        np.array where each entry is a separate entry to the logfile
    
    """
    # List all logfiles
    date_logfiles = list(glob(f"{data_dir}/{recorder}/loclog.txt"))
    try:
        assert(len(date_logfiles) >= 1)
    except:
        warnings.warn(f"No loclog found for recorder {recorder} and date {date}")
        return []

    # Read contents of all logfiles
    write_time_vals = []
    for logfile in date_logfiles:
        write_time_vals.append(pd.read_csv(logfile, sep='\t', encoding="unicode_escape").values)
    return np.concatenate(write_time_vals)


def estimate_typical_write_speed_microseconds(write_times, num_samples=30):
    return int(mode([(write_times[idx+1]-write_times[idx]).microseconds for idx in range(num_samples)]).mode)

def get_buffer_insert_lengths(write_times, sample_rate=44100):
    """Finds positions and lengths of missing buffers
    
    Finds the typical gap between write times (usually ~185770 microseconds),
    typical_write_microsec
    
    Compares each write time to the expected write time. If the difference between
    write times is more than 10 microseconds, then a buffer overflow is assumed.
    These differences in write times are usually approximately multiples of 
    typical_write_microsec but sometimes are not. Unsure if these are true missing buffers.
    
    When a buffer is missing, calculates the sample index of the missing buffer and the
    number of samples that should be added at that index. These are estimated 
    based on the input sample_rate.
    
    An example which explains the buffer indexing and length logic: 
        ```
        true_write_time was 6 blocks
        expected_write_time was 3 blocks
        first_write_time was 1 block
        start_time was 0 blocks

        at first_write_time, inserted block from start_time to first_write_time
        at second_write_time, inserted blcok from first_write_time to second_write_time
        at third_write_time, missed a block that would have been inserted at 
            second_write_time to expected_write_time
        at fourth_write_time, missed a block
        at fifth_write_time, missed a block
        at sixth_write_time, inserted a block at the position of second_write_time. 
            these were the true samples from fifth_write_time to sixth_write_time.

        To correct this, we need to add:
        0s for second_write_time to expected_write_time (third_write_time)
        0s for third_write_time to fourth_write_time
        0s for fourth_write_time to fifth_write_time

        at fifth_write_time, it's got it covered again.

        so insert
        block_length_per_second_in_samples * (true_write_time - expected_write_time)
        at time
        (expected_write_time-first_write_time)*block_length_per_second_in_samples
        ```
    
    """
    typical_write_microsec = estimate_typical_write_speed_microseconds(write_times, num_samples=100)
    first_write_time = write_times[0]
    
    # How many samples into the recording to insert the buffer?
    # Assume samples will all be inserted into sample array at once,
    # so indices are independent of each other
    overflow_sample_indices = []
    overflow_samples_to_insert = []
    
    # Create an array of expected start times
    expected_write_times = [write_times[0] + datetime.timedelta(microseconds=typical_write_microsec)*x for x in range(len(write_times))]
    
    # If there are overflows, will have to compensate for this in the expected_write_times array
    # e.g. if the sample gets 10s off but never overflows again, 
    # the expected_write_times will be 10s off from that point on.
    # So keep track of this to correct for it in the remainder of expected_write_times
    expected_correction = datetime.timedelta()
    
    num_buffer_overflows_identified=0
    
    # Search for a buffer overrun at each write time
    for true_write_time, expected_write_time in zip(write_times, expected_write_times):
        #print(f"Write time: true={true_write_time}, expected={expected_write_time + expected_correction}. Prev correction: {expected_correction}")
        seconds_of_lag = true_write_time - (expected_write_time + expected_correction)
        #print("seconds_of_lag:", seconds_of_lag)
        
        # If there was a buffer overrun...
        if (seconds_of_lag.days >= 0) & (seconds_of_lag.seconds > 0 or seconds_of_lag.microseconds > 10):
            #print("\n\n\n\n\n\n")
            # The samples came in later than expected = buffer overrun
            # We will have to insert the correct number of samples at the expected time
            # Typically this many samples should be inserted per write: .185769*44100 = 8192
            # These should be inserted at position "true" seconds into the recording (I think?)
            # ==> "true" seconds into recording * 8192
            #print("Diff between true and expected (corrected):", seconds_of_lag)
            #print(f"Write time: true={true_write_time}, expected={expected_write_time + expected_correction}. Prev correction: {expected_correction}")
            seconds_into_sample_array = true_write_time - first_write_time
            sample_index_UNROUNDED = seconds_into_sample_array.seconds * sample_rate + \
                (seconds_into_sample_array.microseconds/1000000) * sample_rate
            sample_index = round(sample_index_UNROUNDED)
            overflow_sample_indices.append(sample_index)
            #print("Index to insert samples to:", sample_index)
            
            num_samples_to_insert_UNROUNDED = seconds_of_lag.seconds * sample_rate + \
                (seconds_of_lag.microseconds/1000000)*sample_rate
            num_samples_to_insert = round(num_samples_to_insert_UNROUNDED)
            overflow_samples_to_insert.append(num_samples_to_insert)
            #print("Number of samples to insert:", num_samples_to_insert)
            
            overflow_amount = abs(expected_write_time-true_write_time).seconds*1000000 + \
                abs(expected_write_time-true_write_time).microseconds
            #print("Seconds missed so far:", overflow_amount/typical_write_microsec)
            
            num_buffer_overflows_identified += (
                seconds_of_lag.seconds*1000000 + seconds_of_lag.microseconds)/typical_write_microsec
            #print("Buffers unwritten so far:", x)
            
        expected_correction = expected_correction + seconds_of_lag
        #print("New expected correction:",expected_correction)
        #print("\n\n\n")
        #print("Num buffer overflows identified:",num_buffer_overflows_identified)

    #return num_buffer_overflows_identified
    return overflow_sample_indices, overflow_samples_to_insert

def calculate_resample_array(start, end, samples, sample_rate):
    """
    Arguments:
        start: datetime.datetime for exact start (including with microseconds)
        end: datetime.datetime for exact end (including with microseconds)
        samples: audio sample array
        sample_rate: desired output audio sample rate
        
    Returns:
        t, locations of where to interpolate within the array
    """
    num_samples_taken = len(samples)

    # If the recording were perfect, its duration and number of samples would be
    ideal_duration = end - start
    ideal_duration_sec = ideal_duration.seconds + ideal_duration.microseconds/1000000
    ideal_num_samples = sample_rate * ideal_duration_sec

    # What gap between the samples do we need to correct for the recorder's imperfections
    sample_spacing = num_samples_taken/ideal_num_samples

    # Create an array of times the samples should be interpolated to
    t = np.arange(
        start=0,
        stop=num_samples_taken,
        step=sample_spacing
    )
    
    #print("number of samples taken originally:",num_samples_taken)
    #print("ideal number of samples:", ideal_num_samples)
    #print()
    #print("number of samples to resample to:",len(t))
    #print("highest number in resample array:",max(t))

    # Make sure we don't sample past the end of the data we actually have
    assert max(t) < num_samples_taken
    
    return t


def insert_missing_buffers(
    recorder,
    date,
    time,
    data_dir,
    overflow_sample_indices,
    overflow_samples_to_insert,
    sample_rate=44100
):
    """Insert 0s at buffer overrun locations
    
    Loads audio from a particular recorder, date, and time.
    Accounts for missing buffers by inserting arrays of 0s of the
    correct length within the audio's sample array.
    
    Arguments:
        recorder (string): recorder name (e.g. "A1", "G7")
        date (string): date of recording in format YYYYMMDD e.g. "20230628"
        time (string): start time of recording in format HHMM, e.g. "0929"
        data_dir (string): location of original data files
        overflow_sample_indices (np.array): list of the indices of samples where
            overflows should be inserted. Note, these are the indices *in the 
            original array*, i.e. don't account for changes to the array after
            overflow samples are added in
        overflow_samples_to_insert (np.array): number of 0s to insert at each
            index given by overflow_sample_indices. These arrays must exactly match
            e.g. zip(overflow_sample_indices, overflow_samples_to_insert) pairs the
            index with the appropriate number of samples to insert
            
    """
    # Get filename of recording
    filename = get_recording_path(recorder=recorder, date=date, hour_minute=time, data_dir=data_dir)
    
    # Load audio from filename
    a = Audio.from_file(filename)
    
    # Convert to samples list for improved speed
    original_sample_array = a.samples
    corrected_sample_array = np.array([])
    #original_sample_list = original_sample_array.tolist()
    #corrected_sample_list = []
    
    # Add correct samples and buffer correction 0s to new list iteratively
    prev_idx = 0
    for idx, num_samples in zip(overflow_sample_indices, overflow_samples_to_insert):
        corrected_sample_array = np.concatenate(
            [corrected_sample_array, original_sample_array[prev_idx:idx],[0]*num_samples])
        prev_idx = idx
    # Then fill in the remaining samples from the original array
    corrected_sample_array = np.concatenate(
        [corrected_sample_array, original_sample_array[prev_idx:]])
    
    return Audio(corrected_sample_array, sample_rate=sample_rate)
    


def interpolate_audio(start, end, audio):
    """Interpolate audio using given start and end times
    
    Arguments:
        start (datetime.datetime): recorder-provided start time for the recording
        end (datetime.datetime): recorder-provided end time for the recording
        audio (opensoundscape.audio.Audio): Audio instance of recording
    
    Returns:
        opensoundscape.audio.Audio object of resampled audio
    """
    t = calculate_resample_array(
        start,
        end,
        audio.samples,
        audio.sample_rate)
    
    array, t = resample(x=audio.samples, num=len(t), t=t)
    
    return Audio(array, sample_rate=audio.sample_rate)








### LOCALIZATION UTILITIES ###

def distance(x1,x2,y1,y2):
    """Calculate Euclidean distance in 2D
    """
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def get_closest_recorders(
    playback_id,
    playback_coords, 
    aru_coords,
    num_recorders=4, 
    x=None,
    y=None,
):
    """Get coordinates of recorders closest to a playback_id
    or other location
    
    Arguments:
        playback_id (string): "PB" + a number 1-26.
            pass None if using your own coordinates
        playback_coords (pandas DataFrame): coordinates of 
            playbacks with index=playback_id and columns x and y.
            Pass None if using your own coordinates.
        aru_coords (pandas DataFrame): dataframe of coordinates 
            of autonomous recorders. Its index must be the names 
            of the recorders and it needs to contain columns x and y
        num_recorders (integer): number of top closest recorders 
            to return. Default is to return 4 recorders.
        
    """
    if playback_id is None:
        try:
            assert type(x) == float
            assert type(y) == float
        except AssertionError:
            raise ValueError("If playback_id is not provided, need to provide x and y coordinates")
        print(f"Closest recorders to ({x}, {y}):")
    else:
        # Get coords of playback
        playback_name = "PB" + str(playback_id)
        pb_x = playback_coords.x[playback_name]
        pb_y = playback_coords.y[playback_name]
        print(f"Closest recorders to {playback_name}:")
    
    # Create a dataframe of recorder distances
    recorder_distances = {}
    for idx, row in aru_coords.iterrows():
        recorder_distances[idx] = distance(pb_x, row.x, pb_y, row.y)

    rec_distances = pd.Series(recorder_distances).sort_values()[:num_recorders]
    print(rec_distances)
    return list(rec_distances.index)

def get_latest_start_second(filenames):
    """Find the the first second all recorders were recording
    
    From a set of filenames, determine the start times of the 
    filenames and figure out the time that the latest recorder
    started recording. This can be used to trim recordings
    so that they all start at the same time.
    """
    starts=[]
    ends=[]
    for filename in filenames:
        start, end = extract_start_end(filename.name) #the 3: index removes the recorder name and underscore
        starts.append(start)
        ends.append(end)
    
    latest_start = max(starts)
    next_second = latest_start + datetime.timedelta(seconds=1) - datetime.timedelta(microseconds=latest_start.microsecond)
    return next_second


def get_earliest_end_second(filenames):
    """Find the the last second all recorders were recording
    
    From a set of filenames, determine the end times of the 
    filenames and figure out the time that the earliest recorder
    stopped recording. This can be used to trim recordings
    so that they all end at the same time.
    """
    starts=[]
    ends=[]
    for filename in filenames:
        start, end = extract_start_end(filename.name) #the 3: index removes the recorder name and underscore
        starts.append(start)
        ends.append(end)
        
    earliest_end = min(ends)
    previous_second = earliest_end - datetime.timedelta(microseconds=earliest_end.microsecond)
    return previous_second


def get_audio_from_time(clip_start, clip_length_s, original_start, original_audio):
    """
    Arguments:
        clip_start (datetime.timedelta): start time of the desired clip
        clip_length_s: how long clip should be
        original_start (datetime.timedelta): original start
        original_audio: the audio file to extract sound from
    """
    assert clip_start > original_start
    sec_into_original = (clip_start - original_start).seconds + (clip_start - original_start).microseconds/1000000
    return original_audio.trim(sec_into_original, sec_into_original + clip_length_s)
