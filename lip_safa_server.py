from uuid import uuid4
from animation_flow import make_image_animation_dataflow
from w2l.utils.face_detect import detect_face_and_dump_from_video
from w2l.utils.generate import generate_video
import torch
import time
import os
from pathlib import Path

from utils.mysql_dbtool import dbtools
from utils.gcs_tool import upload_to_gcs, download_gcs


def check_audio(gcs_path, audio_dir):
    audio_path = os.path.join(audio_dir, os.path.basename(gcs_path))
    if not os.path.exists(audio_path):
        download_gcs(gcs_path, audio_dir)
    return audio_path


def worker(data_dir):
    FACE_DETECT_BATCH_SIZE = 4 if os.environ.get(
        "FACE_DETECT_BATCH_SIZE") is None else int(os.environ.get("FACE_DETECT_BATCH_SIZE"))
    GENERATE_BATCH_SIZE = 32 if os.environ.get(
        "GENERATE_BATCH_SIZE") is None else int(os.environ.get("GENERATE_BATCH_SIZE"))

    preprocess_dir = f'{data_dir}/preprocess'
    audio_dir = f'{data_dir}/audio'
    os.makedirs('temp', exist_ok=True)
    os.makedirs(preprocess_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    with dbtools.session() as sess:
        while True:
            try:
                job = dbtools.get_job_join()
                time.sleep(5)
                if job is not None:
                    st = time.time()
                    if dbtools.set_ticket_job(sess.processing_ticket_id, job['id']):
                        tempdir = ".".join(os.path.basename(
                            job["video_path"]).split(".")[:-1])
                        dumpdir = os.path.join(preprocess_dir, tempdir)
                        face_config = os.path.join(dumpdir, 'face.tsv')
                        dbtools.update_job_process_datetime(job['id'], True)
                        if not os.path.exists(face_config):
                            dbtools.update_job_progress(
                                job['id'], 'preprocessing', 25)
                            face_config = detect_face_and_dump_from_video(
                                job['video_path'], dumpdir, 'cuda', 96, face_detect_batch_size=FACE_DETECT_BATCH_SIZE, smooth=True)
                            torch.cuda.empty_cache()
                        audio_path = check_audio(job['audio_path'], audio_dir)
                        dbtools.update_job_progress(
                            job['id'], 'generate lipsync', 50)
                        generate_video(face_config, audio_path, 'ckpt/wav2lip_gan.pth',
                                       '/tmp/lip.mp4', batch_size=GENERATE_BATCH_SIZE)
                        # image_name = Path(job['image_filename']).stem
                        # video_name = Path(job['video_filename']).stem
                        # audio_name = Path(job['audio_filename']).stem
                        torch.cuda.empty_cache()
                        filename = uuid4().hex
                        result_path = f"/tmp/{filename}.mp4"
                        gcs_path = f"result/{filename}.mp4"
                        result_filename = os.path.basename(result_path)
                        image_content = job['image_content']
                        dbtools.update_job_progress(
                            job['id'], 'generate image animation', 75)
                        make_image_animation_dataflow(
                            image_content, '/tmp/lip.mp4', result_path, 'ckpt', crf=job['out_crf'], use_crop=True, use_gfp=job['enhance'])
                        torch.cuda.empty_cache()
                        dbtools.update_job_progress(job['id'], 'finish', 100)
                        dbtools.update_job_result(
                            job['id'], result_filename, gcs_path)
                        dbtools.update_job_process_datetime(job['id'], False)
                        upload_to_gcs(result_path, gcs_path)
                        et = time.time()
                        cost_time = f'{et-st:0.2f}'
                        print(cost_time)
                    else:
                        continue
            except Exception as e:
                print(e)


if __name__ == '__main__':

    worker('datadir')
