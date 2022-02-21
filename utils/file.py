from uuid import uuid4
from utils.mysql_dbtool import dbtools
from datetime import datetime
import os
from pathlib import Path
import cv2
import subprocess
import hashlib
from utils.gcs_tool import upload_to_gcs, get_blob_list, delete_blobs, download_gcs


def get_size_mb(inputfile):
    size = os.stat(inputfile).st_size
    return size/(1000*1000)


def get_fps(inputfile):
    video = cv2.VideoCapture(inputfile)
    fps = video.get(cv2.CAP_PROP_FPS)
    return fps


def get_duration(inputfile):
    result = subprocess.Popen('ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {}'.format(inputfile),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    output = result.communicate()
    duration = float(output[0].decode())
    return duration


def get_file_md5(inputfile):
    f = open(inputfile, "rb")
    md5 = hashlib.md5(f.read())
    f.close()
    return md5.hexdigest()


def add_video2db(client_id, filepath, comment):
    filename = Path(filepath).stem
    filepath_gcs = os.path.join('video', os.path.basename(filepath))
    upload_datetime = datetime.now()
    expiration_datetime = None
    size_mb = get_size_mb(filepath)
    duration = get_duration(filepath)
    fps = get_fps(filepath)
    video_md5 = get_file_md5(filepath)
    data = (None, client_id, filename, filepath_gcs, video_md5, upload_datetime,
            expiration_datetime, size_mb, duration, fps, comment)
    if dbtools.check_md5('video', video_md5) is None:
        dbtools.insert_video(data)
        upload_to_gcs(filepath, filepath_gcs)


def add_audio2db(client_id, filepath, comment):
    filename = Path(filepath).stem
    filepath_gcs = os.path.join('audio', uuid4().hex+'.wav')
    upload_datetime = datetime.now()
    expiration_datetime = None
    size_mb = get_size_mb(filepath)
    duration = get_duration(filepath)
    audio_md5 = get_file_md5(filepath)
    data = (None, client_id, None, filename, filepath_gcs, audio_md5, upload_datetime,
            expiration_datetime, size_mb, duration, comment)
    if dbtools.check_md5('audio', audio_md5) is None:
        dbtools.insert_audio(data)
        upload_to_gcs(filepath, filepath_gcs)


def add_image2db(client_id, filepath, filepath2, gender, comment):
    filename = Path(filepath).stem
    with open(filepath, 'rb') as f:
        content = f.read()
    with open(filepath2, 'rb') as f:
        display_content = f.read()
    md5 = get_file_md5(filepath)
    upload_datetime = datetime.now()
    size_mb = get_size_mb(filepath)
    data = (None, client_id, gender, filename, display_content, content,
            md5, upload_datetime, size_mb, comment)
    dbtools.insert_image(data)


def add_job(client_id, image_id, video_id, audio_id, out_crf, enhance, comment=''):
    data = (None, client_id, image_id, video_id, audio_id, None, None, 'init',
            0, datetime.now(), None, None, out_crf, enhance, comment)
    try:
        dbtools.insert_job(data)
    except Exception as e:
        print(e)


def add_client(account):
    data = (None, account, uuid4().hex, uuid4().hex, datetime.now(), '')
    try:
        dbtools.insert_client(data)
    except Exception as e:
        print(e)
