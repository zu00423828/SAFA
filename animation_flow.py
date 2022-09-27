import cv2
import numpy as np
import os
import torch
from animation_demo import create_image_animation
# import face_alignment
import subprocess
from tqdm import trange
from utils.crop_video import process_video
# from gpen.face_enhancement import FaceEnhancement
from gpen.face_model.face_gan import FaceGAN


def concat_video(left, right, out_path):
    video1 = cv2.VideoCapture(left)
    video2 = cv2.VideoCapture(right)
    fps = video1.get(5)
    out = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (512, 256))
    while video1.isOpened():
        ret, frame1 = video1.read()
        if not ret:
            break
        ret, frame2 = video2.read()
        if not ret:
            break
        frame = np.concatenate([frame1, frame2], axis=1)
        out.write(frame)
    video1.release()
    video2.release()
    out.release()
    # command=f'ffmpeg -i /tmp/temp_concat.mp4 -i /tmp/temp.wav  {out_path}'
    # subprocess.call(command,shell=True)


def check(dir):
    source_video = cv2.VideoCapture(f'{dir}/source.mp4')
    result_video = cv2.VideoCapture(f'{dir}/driving.mp4')
    out = cv2.VideoWriter(
        f'{dir}/check.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (256, 256))
    while source_video.isOpened():
        ret, frame1 = source_video.read()
        if not ret:
            break
        ret, frame2 = result_video.read()
        if not ret:
            break
        frame = (frame1/2)+(frame2/2)
        frame = frame.astype(np.uint8)
        out.write(frame)
    out.release()
    source_video.release()
    result_video.release()


def video_gpen_process(origin_video_path, model_dir, out_video_path='/tmp/paste_temp.mp4'):
    # processer = FaceEnhancement(base_dir=model_dir, in_size=512, model='GPEN-BFR-512', sr_scale=2,
    #                             use_sr=False, sr_model=None)
    model = FaceGAN(model_dir, 512, None, 'GPEN-BFR-512',
                    2, 1, None, device='cuda')
    full_video = cv2.VideoCapture(origin_video_path)
    h = int(full_video.get(4))
    w = int(full_video.get(3))
    fps = full_video.get(5)
    frame_count = int(full_video.get(7))
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(
        *'XVID'), fps, (w, h))
    for _ in trange(frame_count):
        _, frame = full_video.read()
        # img_out, _, _ = processer.process(
        #     frame, aligned=False)
        # img_out = cv2.resize(img_out, (w, h))
        img_out = model.process(frame)
        img_out = cv2.resize(img_out, (w, h))
        # mask = np.zeros([h+2, w+2], np.uint8)
        # cv2.floodFill(img_out, mask, (0, 0), (255, 255, 255), (
        #     50, 50, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
        out_video.write(img_out)
    # print(origin_video_path, out_video_path)
    return out_video_path


def make_image_animation_dataflow(source_path, driving_origin_path, result_path, model_dir, use_crop=False, crf=0, use_gfp=True, use_best=False, face_data=None, pre_enhance=False):
    config_path = f"{os.path.split(os.path.realpath(__file__))[0]}/config/end2end.yaml"
    if use_crop:
        print('crop driving video', flush=True)
        driving_video_path = process_video(
            driving_origin_path, '/tmp/driving.mp4', min_frames=15, face_data=face_data, increase=-0.1)
        torch.cuda.empty_cache()
    else:
        driving_video_path = driving_origin_path
    command = f"ffmpeg -y -i {driving_video_path} /tmp/temp.wav "
    subprocess.call(command, shell=True)
    if pre_enhance:
        driving_video_path = video_gpen_process(
            driving_video_path, model_dir, out_video_path='/tmp/driving_enhace.mp4')
    print('create animation', flush=True)
    safa_model_path = f'{model_dir}/final_3DV.tar'
    safa_video = create_image_animation(source_path, driving_video_path, '/tmp/temp.mp4', config_path,
                                        safa_model_path, with_eye=False, relative=True, adapt_scale=True, use_best_frame=use_best)  # with_eye=False 可解決單眼扎眼
    torch.cuda.empty_cache()
    # print('extract landmark', flush=True)
    # ldmk_path = extract_landmark(safa_video, '/tmp/ldmk.pkl')
    # print('blur mouth video', flush=True)
    # torch.cuda.empty_cache()
    # safa_video = blur_video_mouth(
    #     safa_video, ldmk_path, '/tmp/blur.avi', kernel=3)
    print('enhaance process', flush=True)
    # paste_video_path = video_gfpgan_process(
    #     safa_video, ldmk_path, use_gfp, model_dir=model_dir)
    paste_video_path = video_gpen_process(safa_video, model_dir)
    torch.cuda.empty_cache()
    # -preset veryslow
    command = f"ffmpeg -y -i {paste_video_path} -i /tmp/temp.wav  -crf  {crf} -vcodec h264  {result_path} "
    # command = f"ffmpeg   -y -i {paste_video_path} -i /tmp/temp.wav -crf {crf} -vcodec h264_nvenc  {result_path}"
    subprocess.call(command, shell=True)


if __name__ == '__main__':

    from mock import generate_lip_video
    from pathlib import Path
    from glob import glob
    import time
    root = '/home/yuan/hdd/09_19'
    face_data = '/home/yuan/hdd/driving_video/model2-crop-wav2lip/face.pkl'
    print("root", root)
    job_list = ['BSD-Series-new/中文', 'BSD-Series-new/英文',
                '04-優選鋸', '耀登', '曙光/DN-Series', '曙光/DN-Series-old']
    for job_item in job_list:
        print(job_item)
        for audio_path in sorted(glob(f'{root}/audio/{job_item}/*wav')):
            st = time.perf_counter()
            image_input = f"{root}/img/0429_1-ok.png"
            lip_dir = f'{root}/lip/{job_item}'
            os.makedirs(lip_dir, exist_ok=True)
            lip_path = os.path.join(lip_dir, Path(
                audio_path).stem.replace(' ', '-')+'.mp4')
            if not os.path.exists(lip_path):
                generate_lip_video(
                    face_data, audio_path, lip_path)
                torch.cuda.empty_cache()
            save_dir = os.path.join(root, Path(
                image_input).parent.name+'_out', job_item)
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, 'result_' +
                                    Path(lip_path).stem+'.mp4')
            if os.path.exists(out_path):
                continue
            try:
                make_image_animation_dataflow(
                    image_input, lip_path, out_path, 'ckpt/', use_crop=False, face_data=face_data)
                torch.cuda.empty_cache()
            except Exception as e:
                print(e)
            print('cost:', time.perf_counter()-st)

    root = '/home/yuan/hdd/09_22'
    face_data = '/home/yuan/hdd/driving_video/model2-crop-wav2lip/face.pkl'
    lip_path = f'{root}/lip/en_05.mp4'
    print("root", root)

    for img_path in sorted(glob(f'{root}/img/western/*g')):
        st = time.perf_counter()
        save_dir = os.path.join(root, Path(
            image_input).parent.name+'_out', 'western')
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, 'result_' +
                                Path(img_path).stem+'.mp4')
        if os.path.exists(out_path):
            continue
        try:
            make_image_animation_dataflow(
                img_path, lip_path, out_path, 'ckpt/', use_crop=False, face_data=face_data)
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
        print('cost:', time.perf_counter()-st)

    root = '/home/yuan/hdd/09_26'
    face_data = '/home/yuan/hdd/driving_video/model2-crop-wav2lip/face.pkl'
    # face_data = '/home/yuan/hdd/driving_video/model2-crop-ebt/face.pkl'
    print("root", root)

    for audio_path in sorted(glob(f'{root}/audio/*wav')):
        st = time.perf_counter()
        image_input = f"{root}/img/1024x1024-0926ok.jpg"
        lip_dir = f'{root}/lip/'
        os.makedirs(lip_dir, exist_ok=True)
        lip_path = os.path.join(lip_dir, Path(
            audio_path).stem.replace(' ', '-')+'.mp4')
        if not os.path.exists(lip_path):
            generate_lip_video(
                face_data, audio_path, lip_path)
            torch.cuda.empty_cache()
        save_dir = os.path.join(root, Path(
            image_input).parent.name+'_out', 'new')
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, 'result_' +
                                Path(lip_path).stem+'.mp4')
        if os.path.exists(out_path):
            continue
        try:
            make_image_animation_dataflow(
                image_input, lip_path, out_path, 'ckpt/', use_crop=False, face_data=face_data)
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
        print('cost:', time.perf_counter()-st)

    root = '/home/yuan/hdd/09_16'
    face_data = '/home/yuan/hdd/driving_video/model2-crop-wav2lip/face.pkl'
    print("root", root)
    job_list = ['asia', 'us']
    for job_item in job_list:
        print(job_item)
        for image_input in sorted(glob(f'{root}/img/{job_item}/*g')):
            st = time.perf_counter()
            # image_input = f"{root}/img/0429_1-ok.png"
            lip_dir = f'{root}/lip/'
            lip_path = os.path.join(lip_dir, job_item+'.mp4')
            save_dir = os.path.join(root, 'img'+'_out', job_item)
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, 'result_' +
                                    Path(image_input).stem+'.mp4')
            if os.path.exists(out_path):
                continue
            try:
                make_image_animation_dataflow(
                    image_input, lip_path, out_path, 'ckpt/', use_crop=False, face_data=face_data)
                torch.cuda.empty_cache()
            except Exception as e:
                print(e)
            print('cost:', time.perf_counter()-st)
