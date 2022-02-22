from logging import root
import pickle
import cv2
import numpy as np
import os
import torch
from animation_demo import create_video_animation, create_image_animation
from utils.face_restore_helper import create_face_helper
from utils.mask import _cal_mouth_contour_mask
from utils.blend import LaplacianBlending
import face_alignment
import subprocess
from tqdm import trange
from utils.crop_video import process_video
from gfpgan import GFPGANer


def concat_video(left, right, out_path):
    video1 = cv2.VideoCapture(left)
    video2 = cv2.VideoCapture(right)
    out = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (512, 256))
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


def extract_landmark(video_path, out_path):

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D)
    video = cv2.VideoCapture(video_path)
    all_landmarks = []
    frame_count = int(video.get(7))
    for _ in trange(frame_count):
        ret, frame = video.read()
        if not ret:
            break
        result = fa.get_landmarks(frame)
        if result is None:
            landmark = None
        else:
            landmark = result[0]
        all_landmarks.append(landmark)

    with open(out_path, 'wb') as f:
        pickle.dump(all_landmarks, f)
    return out_path


def face_preprocess(input_video, landmark_file, save_name):
    face_helper = create_face_helper(512)
    video = cv2.VideoCapture(input_video)
    f = open(landmark_file, 'rb')
    video_landmarks = pickle.load(f)
    out_path = f'{save_name}.mp4'
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(
        *'MP4V'), 30.0, (256, 256))
    matrix_list = []
    new_landmark_list = []
    data_len = min(len(video_landmarks), int(video.get(7)))
    for i in trange(data_len):
        _, frame = video.read()
        frame_landmark = video_landmarks[i]
        if frame_landmark is None:
            matrix_list.append(None)
            new_landmark_list.append(None)
            large_face = np.zeros((256, 256, 3), np.uint8)
        else:
            face_helper.clean_all()
            face_helper.read_image(frame)
            face_helper.get_face_landmarks_3(
                frame_landmark, only_keep_largest=True, eye_dist_threshold=5)
            face_helper.align_warp_face()

            assert len(face_helper.cropped_faces) != 0

            affine_matrix = face_helper.affine_matrices[0]
            large_face = face_helper.cropped_faces[0]
            affine_landmarks = (
                np.matmul(affine_matrix[:, :2], frame_landmark.T) + affine_matrix[:, 2:]).T
            matrix_list.append(affine_matrix)
            new_landmark_list.append(affine_landmarks)
        out.write(cv2.resize(large_face, (256, 256)))
    out_data_path = f'{save_name}_data.pkl'
    with open(out_data_path, 'wb') as f:
        pickle.dump(
            {'matrix': matrix_list, 'affine_landmark': new_landmark_list}, f)
    return out_path, out_data_path


def create_lb(iters: int = None, ksize: int = 3, sigma=0):
    lb = LaplacianBlending(
        sigma=sigma,
        ksize=ksize,
        iters=4 if iters is None else iters).eval()
    for param in lb.parameters():
        param.requires_grad = False
    return lb


def quantize_position(x1, x2, y1, y2, iters=None):
    w = x2 - x1
    h = y2 - y1
    x_center = (x2 + x1) // 2
    y_center = (y2 + y1) // 2
    half_w = np.math.ceil(w / (2 ** iters)) * \
        (2 ** (iters - 1))
    half_h = np.math.ceil(h / (2 ** iters)) * \
        (2 ** (iters - 1))
    x1 = x_center - half_w
    x2 = x_center + half_w
    y1 = y_center - half_h
    y2 = y_center + half_h

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    return int(x1), int(x2), int(y1), int(y2)


def paste_origin_video(source_origin_path, safa_video_path, temp_dir, landmark_path, source_video_data_path):
    full_video = cv2.VideoCapture(source_origin_path)
    crop_video = cv2.VideoCapture(safa_video_path)
    out_video_path = f'{temp_dir}/paste_temp.mp4'
    h = int(full_video.get(4))
    w = int(full_video.get(3))
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(
        *'XVID'), 30.0, (w, h))
    lb = create_lb(4)
    with open(landmark_path, 'rb') as f:
        source_landmark = pickle.load(f)
    with open(source_video_data_path, 'rb') as f:
        source_data = pickle.load(f)
        soucre_matrix = source_data['matrix']
        # source_affine_landmark=source_data['affine_landmark'] #512*512  landmark
    # min(len(source_affine_landmark),len(drive_data_affine_landmark))
    frame_count = int(crop_video.get(7))
    for i in trange(frame_count):
        _, full_frame = full_video.read()
        _, crop_frame = crop_video.read()

        if source_landmark[i] is None:
            full_out = full_frame.copy()
        else:
            crop_frame = cv2.resize(crop_frame, (512, 512))
            inv_matrix = cv2.invertAffineTransform(soucre_matrix[i])
            new_frame = cv2.warpAffine(
                crop_frame, inv_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))

            x1, x2, y1, y2 = quantize_position(0, w, 0, h, 4)
            mask = _cal_mouth_contour_mask(
                source_landmark[i], y2, x2, None, 0.1)
            if x2 > w or y2 > h:
                full_frame = cv2.copyMakeBorder(full_frame, 0, max(
                    0, y2-h), 0, max(0, x2-w), cv2.BORDER_CONSTANT, value=[255, 255, 255])
                new_frame = cv2.copyMakeBorder(new_frame, 0, max(
                    0, y2-h), 0, max(0, x2-w), cv2.BORDER_CONSTANT, value=[255, 255, 255])
            mask_tesor = torch.tensor(
                mask, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            y_tensor = torch.tensor(
                full_frame/255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            x_tensor = torch.tensor(
                new_frame/255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            out = lb(y_tensor, x_tensor, mask_tesor)
            full_out = (out[0][:, :h, :w].permute(
                1, 2, 0)*255).numpy().astype(np.uint8)
        out_video.write(full_out)
    out_video.release()
    # out_video2.release()
    full_video.release()
    crop_video.release()
    return out_video_path


def video_gfpgan_process(origin_video_path, landmark_path, use_gfp=True, model_dir='ckpt'):
    if use_gfp:
        restorer = GFPGANer(
            model_path=f'{model_dir}/GFPGANCleanv1-NoCE-C2.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None)

    full_video = cv2.VideoCapture(origin_video_path)
    out_video_path = '/tmp/paste_temp.mp4'
    h = int(full_video.get(4))
    w = int(full_video.get(3))
    fps = full_video.get(5)
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(
        *'XVID'), fps, (w, h))
    lb = create_lb(4)
    with open(landmark_path, 'rb') as f:
        source_landmark = pickle.load(f)
    # min(len(source_affine_landmark),len(drive_data_affine_landmark))
    frame_count = int(full_video.get(7))
    for i in trange(frame_count):
        _, full_frame = full_video.read()
        if source_landmark[i] is None:
            full_out = full_frame.copy()
        else:
            enhance_img = full_frame.copy()
            if use_gfp:
                _, _, enhance_img = restorer.enhance(
                    full_frame, has_aligned=False, only_center_face=False, paste_back=True)
                enhance_img = cv2.resize(
                    enhance_img, (full_frame.shape[1], full_frame.shape[0]))
            x1, x2, y1, y2 = quantize_position(0, w, 0, h, 4)
            mask = _cal_mouth_contour_mask(
                source_landmark[i], y2, x2, None, 0.1)
            if x2 > w or y2 > h:
                full_frame = cv2.copyMakeBorder(full_frame, 0, max(
                    0, y2-h), 0, max(0, x2-w), cv2.BORDER_CONSTANT, value=[255, 255, 255])
                enhance_img = cv2.copyMakeBorder(enhance_img, 0, max(
                    0, y2-h), 0, max(0, x2-w), cv2.BORDER_CONSTANT, value=[255, 255, 255])
            mask_tesor = torch.tensor(
                mask, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            y_tensor = torch.tensor(
                full_frame/255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            x_tensor = torch.tensor(
                enhance_img/255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            out = lb(y_tensor, x_tensor, mask_tesor)
            full_out = (out[0][:, :h, :w].permute(
                1, 2, 0)*255).numpy().astype(np.uint8)
        out_video.write(full_out)
    out_video.release()
    # out_video2.release()
    full_video.release()
    return out_video_path


def make_animation_dataflow(source_origin_path, driving_origin_path, temp_dir, result_path, model_path, config_path=None, add_audo=False):
    '''
    參數  
    source_origin_path：被操控的原影片路徑 \n
    driving_origin_path：操控的原影片路徑 \n
    temp_dir：暫存用的資料夾 \n
    result_path：操控後的影片路徑 \n
    model_path：SAFA模型的路徑 \n
    config_path=None：SAFA模型的config預設是None，可以直接省略 \n
    add_audo=False：是否要增加聲音，預設FALSE \n
    '''
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if config_path is None:
        config_path = f"{os.path.split(os.path.realpath(__file__))[0]}/config/end2end.yaml"
    print('extract_landmark: source')
    landmark_path = extract_landmark(
        source_origin_path, f'{temp_dir}/source.pkl')
    print('crop_video: source')
    source_video_path, source_data_path = face_preprocess(
        source_origin_path, landmark_path, f'{temp_dir}/source')
    print('crop_video: driving')
    driving_video_path, driving_video_data_path = face_preprocess(
        driving_origin_path, landmark_path, f'{temp_dir}/driving')
    print('generate safa_result_video')
    safa_video_path = create_video_animation(
        source_video_path, driving_video_path, f'{temp_dir}/temp.mp4', config_path, model_path, with_eye=True, relative=False, adapt_scale=True)
    print('paste safa_result_video on the origin source video ')
    temp_paste_video_path = paste_origin_video(
        source_origin_path, safa_video_path, temp_dir, landmark_path, source_data_path)
    video = cv2.VideoCapture(temp_paste_video_path)
    fps = video.get(5)
    video.release()
    if add_audo:
        command = f"ffmpeg -y -i {driving_origin_path} {temp_dir}/temp.wav "
        subprocess.call(command, shell=True)
        # -preset veryslow
        command = f"ffmpeg -y -i {temp_paste_video_path} -i {temp_dir}/temp.wav -vf fps={fps} -crf 0 -vcodec h264  {result_path} "
        subprocess.call(command, shell=True)
    else:
        # -preset veryslow
        command = f"ffmpeg -y -i {temp_paste_video_path}  -vf fps={fps} -crf 0 -vcodec h264  {result_path} "
        subprocess.call(command, shell=True)


def make_image_animation_dataflow(source_path, driving_origin_path, result_path, model_dir, use_crop=False, crf=0, use_gfp=True,):
    config_path = f"{os.path.split(os.path.realpath(__file__))[0]}/config/end2end.yaml"
    if use_crop:
        print('crop driving video', flush=True)
        driving_video_path = process_video(
            driving_origin_path, '/tmp/driving.mp4', min_frames=15)
        torch.cuda.empty_cache()
    else:
        driving_video_path = driving_origin_path
    print('create animation', flush=True)
    safa_model_path = f'{model_dir}/final_3DV.tar'
    safa_video = create_image_animation(source_path, driving_video_path, '/tmp/temp.mp4', config_path,
                                        safa_model_path, with_eye=True, relative=True, adapt_scale=True, use_best_frame=False)
    torch.cuda.empty_cache()
    print('extract landmark', flush=True)
    ldmk_path = extract_landmark(safa_video, '/tmp/ldmk.pkl')
    torch.cuda.empty_cache()
    print('gfp process', flush=True)
    paste_video_path = video_gfpgan_process(
        safa_video, ldmk_path, use_gfp, model_dir=model_dir)
    command = f"ffmpeg -y -i {driving_video_path} /tmp/temp.wav "
    subprocess.call(command, shell=True)
    # -preset veryslow
    command = f"ffmpeg -y -i {paste_video_path} -i /tmp/temp.wav  -crf  {crf} -vcodec h264  {result_path} "
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    # inference_animation_dataflow('new_test/source_all.mp4','new_test/driving_all.mp4','temp','finish.mp4','ckpt/final_3DV.tar')
    # make_animation_dataflow('test1/1.mp4','test1/1.mp4','test1/temp','finish_t.mp4','ckpt/final_3DV.tar',add_audo=True)
    # make_animation_dataflow('finish.mp4','finish_2/driving_all.mp4','finish_2/temp','finish2.mp4','ckpt/final_3DV.tar',add_audo=True)
    # concat_video('/home/yuan/repo/my_safa/01_18/1.mp4','/home/yuan/repo/my_safa/01_18/out/1_1.mp4','concat.mp4')

    # root='/home/yuan/hdd/safa_test/01_18_2'
    # make_image_animation_dataflow(f'{root}/EP010-08.jpg',f'{root}/1.mp4',f'{root}/1_gfpgan.mp4','ckpt/final_3DV.tar',use_crop=False)
    # concat_video(f'{root}/1_gfpgan.mp4',f'{root}/out/1.mp4','concat2.mp4')

    # root = '/home/yuan/hdd/safa_test/02_19/test'
    # driving_video_path = os.path.join(
    #     root, 'driving_man-female-tw-long-120k-0-.mp4')
    # from pathlib import Path
    # from glob import glob
    # # glob(f'{root}/*.png'):
    # for image_path in glob(f'{root}/*.png'):
    #     # image_input = os.path.join(root, image_path)
    #     image_input = image_path
    #     os.makedirs(os.path.join(root, 'out'), exist_ok=True)
    #     out_path = os.path.join(root, 'out', 'result_' +
    #                             Path(image_input).stem+'.mp4')
    #     make_image_animation_dataflow(
    #         image_input, driving_video_path, out_path, 'ckpt/', use_crop=True)
    driving_video = '/home/yuan/share/lip.mp4'
    img = '/home/yuan/hdd/safa_test/02_21/0221_2.jpg'
    make_image_animation_dataflow(
        img, driving_video, '/tmp/out_new.mp4', 'ckpt/', use_crop=True)
    # ffmpeg -i test/input1.mp4  -filter:v "crop=476:476:733:151, scale=256:256" crop.mp4
    # x 733:1209
    # y 151:627
