from uuid import uuid4
from animation_flow import make_image_animation_dataflow
from w2l.utils.face_detect import detect_face_and_dump_from_video
from w2l.utils.generate import generate_video
# from ebt.utils.preprocess import dump_for_inference
# from ebt.scripts.inference import generate_video
import torch
import time
import os
from enum import Enum
from utils.mysql_dbtool import dbtools
from utils.file import add_client, add_video2db
from utils.gcs_tool import upload_to_gcs, download_gcs


class Gender(Enum):
    male = 'male'
    female = 'female'


class Status(Enum):
    init = 'init'
    preprocessing = 'preprocessing'
    lipsyncing = 'lipsyncing'
    image_animating = 'image-animating'
    finished = 'finished'
    error = 'error'
# ENUM('init','preprocessing','lipsyncing','image-animating','finished')


def check_video(gcs_path, video_dir):
    video_path = os.path.join(video_dir, os.path.basename(gcs_path))
    if not os.path.exists(video_path):
        download_gcs(gcs_path, video_dir)
    return video_path


def check_audio(gcs_path, audio_dir):
    audio_path = os.path.join(audio_dir, os.path.basename(gcs_path))
    if not os.path.exists(audio_path):
        download_gcs(gcs_path, audio_dir)
    return audio_path


def lip_process(sess, job, preprocess_dir, video_dir, audio_dir, GENERATE_BATCH_SIZE):
    tempdir = ".".join(os.path.basename(
        job["video_path"]).split(".")[:-1])
    dumpdir = os.path.join(preprocess_dir, tempdir)
    face_config = os.path.join(dumpdir, 'face.pkl')
    dbtools.update_job_process_datetime(job['id'], True)
    video_path = check_video(job['video_path'], video_dir)
    if not os.path.exists(face_config):
        print('preprocessing', flush=True)
        dbtools.update_job_progress(
            job['id'], Status.preprocessing.value, 25)
        face_config = detect_face_and_dump_from_video(
            video_path, dumpdir)
        # face_config = dump_for_inference(
        #     video_path, dumpdir, tdmm_model_path=os.environ['TDMM_MODEL_PATH'])
        torch.cuda.empty_cache()
    audio_path = check_audio(job['audio_path'], audio_dir)
    dbtools.update_job_progress(
        job['id'], Status.lipsyncing.value, 50)
    print('lipsyncing', flush=True)
    generate_video(face_config, audio_path, os.environ['LIP_MODEL_PATH'],
                   '/tmp/finish.mp4', batch_size=GENERATE_BATCH_SIZE)
    torch.cuda.empty_cache()
    filename = uuid4().hex
    result_path = f"/tmp/finish.mp4"
    gcs_path = f"result/{filename}.mp4"
    result_filename = os.path.basename(gcs_path)
    dbtools.update_job_result(
        job['id'], result_filename, gcs_path)
    upload_to_gcs(result_path, gcs_path)
    dbtools.update_job_process_datetime(job['id'], False)
    dbtools.update_job_progress(
        job['id'], Status.finished.value, 100)
    dbtools.update_ticket(sess.processing_ticket_id)
    print(f"job finish {job['id']}", flush=True)


def safa_process(sess, job, preprocess_dir, video_dir, audio_dir, GENERATE_BATCH_SIZE):
    tempdir = ".".join(os.path.basename(
        job["video_path"]).split(".")[:-1])
    dumpdir = os.path.join(preprocess_dir, tempdir)
    face_config = os.path.join(dumpdir, 'face.pkl')
    dbtools.update_job_process_datetime(job['id'], True)
    video_path = check_video(job['video_path'], video_dir)
    if not os.path.exists(face_config):
        print('preprocessing', flush=True)
        dbtools.update_job_progress(
            job['id'], Status.preprocessing.value, 25)
        face_config = detect_face_and_dump_from_video(
            video_path, dumpdir)
        # face_config = dump_for_inference(video_path, dumpdir)
        torch.cuda.empty_cache()
    audio_path = check_audio(job['audio_path'], audio_dir)
    dbtools.update_job_progress(
        job['id'], Status.lipsyncing.value, 50)
    print('lipsyncing', flush=True)
    generate_video(face_config, audio_path, os.environ['LIP_MODEL_PATH'],
                   '/tmp/lip.mp4', batch_size=GENERATE_BATCH_SIZE)
    # generate_video(face_config, audio_path,
    #                output_path='/tmp/lip.mp4', model_path=os.environ['EBT_MODEL_PATH'], exp_model_path=os.environ["EXP_MODEL_PATH"])

    torch.cuda.empty_cache()
    filename = uuid4().hex
    result_path = f"/tmp/finish.mp4"
    gcs_path = f"result/{filename}.mp4"
    result_filename = os.path.basename(gcs_path)
    image_content = job['image_content']
    dbtools.update_job_progress(
        job['id'], Status.image_animating.value, 75)
    make_image_animation_dataflow(
        image_content, '/tmp/lip.mp4', result_path, 'ckpt', crf=job['out_crf'], use_crop=False, use_gfp=job['enhance'], face_data=face_config)
    torch.cuda.empty_cache()
    dbtools.update_job_result(
        job['id'], result_filename, gcs_path)
    upload_to_gcs(result_path, gcs_path)
    dbtools.update_job_process_datetime(job['id'], False)
    dbtools.update_job_progress(
        job['id'], Status.finished.value, 100)
    dbtools.update_ticket(sess.processing_ticket_id)
    print(f"job finish {job['id']}", flush=True)


def worker(data_dir):
    FACE_DETECT_BATCH_SIZE = 4 if os.environ.get(
        "FACE_DETECT_BATCH_SIZE") is None else int(os.environ.get("FACE_DETECT_BATCH_SIZE"))
    GENERATE_BATCH_SIZE = 1 if os.environ.get(
        "GENERATE_BATCH_SIZE") is None else int(os.environ.get("GENERATE_BATCH_SIZE"))

    preprocess_dir = f'{data_dir}/preprocess'
    video_dir = f'{data_dir}/video'
    audio_dir = f'{data_dir}/audio'
    os.makedirs('temp', exist_ok=True)
    os.makedirs(preprocess_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    print('server init', flush=True)
    # init video2 db and upload gcs#
    account = 'share'
    try:
        add_client(account)
    except Exception as e:
        print(e, flush=True)
    account = 'share'
    client_id = dbtools.get_data(
        'client', f"account='{account}'", all=False)['id']
    add_video2db(client_id, 'mock_dir/driving_man.mp4', '')
    add_video2db(client_id, 'mock_dir/driving_woman.mp4', '')

    with dbtools.session() as sess:
        print('ticket_id:', sess.processing_ticket_id, flush=True)
        while True:
            job = None
            try:
                job = dbtools.get_job_join()
                if job is not None:
                    if dbtools.set_ticket_job(sess.processing_ticket_id, job['id']) == True:
                        print('ticket_id: ', sess.processing_ticket_id,
                              ' job_id: ', job['id'], flush=True)
                        if job['image_content'] is None:
                            lip_process(sess, job, preprocess_dir, video_dir,
                                        audio_dir, GENERATE_BATCH_SIZE)
                        else:
                            safa_process(sess, job, preprocess_dir, video_dir,
                                         audio_dir, GENERATE_BATCH_SIZE)
                    else:
                        continue
                else:
                    time.sleep(10)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(e, flush=True)
                comment = job['comment']+str(e)
                dbtools.update_job_error(
                    job['id'], comment, status=Status.error.value)
                dbtools.update_ticket(sess.processing_ticket_id)


if __name__ == '__main__':

    worker('datadir')
