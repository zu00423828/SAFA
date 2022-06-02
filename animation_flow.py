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
from gpen.face_enhancement import FaceEnhancement


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
            model_path=f'{model_dir}/GFPGANv1.3.pth',
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


def blur_video_mouth(video_path, pkl, out_path, kernel=7):
    f = open(pkl, 'rb')
    landmarks = pickle.load(f)
    video = cv2.VideoCapture(video_path)
    h = int(video.get(4))
    w = int(video.get(3))
    fps = video.get(5)
    out_video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(
        *'XVID'), fps, (w, h))
    lb = create_lb(4)
    for i in trange(len(landmarks)):
        _, frame = video.read()
        x1, x2, y1, y2 = quantize_position(0, w, 0, h, 4)
        mask = _cal_mouth_contour_mask(landmarks[i], h, w, None, 0.1)
        mouth = frame.copy()
        blur_mouth = cv2.GaussianBlur(mouth, (kernel, kernel), 0)  # 9*9
        mask_t = torch.tensor(
            mask, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        moth_t = torch.tensor(blur_mouth/255, dtype=torch.float32).permute(
            2, 0, 1).unsqueeze(0)
        origin_t = torch.tensor(frame/255, dtype=torch.float32).permute(
            2, 0, 1).unsqueeze(0)
        # print(origin_t.shape, moth_t.shape, mask_t.shape)
        out = lb(origin_t, moth_t, mask_t, y2, x2)
        full_out = (out[0][:, :h, :w].permute(
            1, 2, 0)*255).cpu().numpy().astype(np.uint8)
        out_video.write(full_out)
    return out_path


def video_gpen_process(origin_video_path, model_dir, out_video_path='/tmp/paste_temp.mp4'):
    processer = FaceEnhancement(base_dir=model_dir, in_size=512, model='GPEN-BFR-512', sr_scale=2,
                                use_sr=False, sr_model=None)
    full_video = cv2.VideoCapture(origin_video_path)
    h = int(full_video.get(4))
    w = int(full_video.get(3))
    fps = full_video.get(5)
    frame_count = int(full_video.get(7))
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(
        *'XVID'), fps, (w, h))
    for _ in trange(frame_count):
        _, frame = full_video.read()
        img_out, _, _ = processer.process(
            frame, aligned=False)
        img_out = cv2.resize(img_out, (w, h))
        out_video.write(img_out)
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


def mouth_teeth(landmarks):
    left_face_width = landmarks[66, 0] - landmarks[60, 0]
    right_face_width = landmarks[64, 0] - landmarks[66, 0]
    delta_left_face_width = left_face_width * 0.1
    delta_right_face_width = right_face_width * 0.1
    delta_face_height = (landmarks[66, 1] -
                         landmarks[62, 1]) * -0.1
    mouth_contours = [[
        [landmarks[60, 0] + delta_left_face_width, landmarks[60, 1]],
        [landmarks[67, 0] + delta_left_face_width, landmarks[67, 1]],
        [landmarks[66, 0], landmarks[66, 1] + delta_face_height],
        [landmarks[65, 0] - delta_right_face_width, landmarks[65, 1]],
        [landmarks[64, 0] - delta_right_face_width, landmarks[64, 1]],
        [landmarks[63, 0] - delta_right_face_width,
            landmarks[63, 1] - delta_face_height],
        [landmarks[62, 0], landmarks[62, 1] - delta_face_height],
        [landmarks[61, 0] + delta_left_face_width,
            landmarks[61, 1] - delta_face_height],
    ]]

    return np.array(mouth_contours, dtype=np.int32)


def mouth_mask(video_path, ldmk_path, out_path):
    f = open(ldmk_path, 'rb')
    landmarks = pickle.load(f)
    video = cv2.VideoCapture(video_path)
    h = int(video.get(4))
    w = int(video.get(3))
    fps = video.get(5)
    out_video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(
        *'XVID'), fps, (w, h))
    lb = create_lb(4)
    for i in trange(len(landmarks)):
        _, frame = video.read()
        mask = np.ones((h, w))
        a = mouth_teeth(landmarks[i])

        mask = cv2.drawContours(
            mask*255, a, -1, (0, 0, 0), -1)
        # mask = cv2.polylines(mask*255, a, True, (0, 0, 0), 3)
        # mask = blur_image(mask, 19)
        # mask = cv2.GaussianBlur(mask, (1, 1), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = np.where(mask < 255, 0, mask)
        frame = np.transpose(frame, (2, 0, 1))
        frame = frame*(mask/255)+(1-mask/255) * \
            np.random.randint(200, 255, size=(3, h, w))
        frame = np.transpose(frame, (1, 2, 0))
        out_video.write(frame.astype(np.uint8))
        cv2.imwrite('/tmp/mask.png', mask)
    return out_path


def make_image_animation_dataflow(source_path, driving_origin_path, result_path, model_dir, use_crop=False, crf=0, use_gfp=True, use_best=False, face_data=None):
    config_path = f"{os.path.split(os.path.realpath(__file__))[0]}/config/end2end.yaml"
    if use_crop:
        print('crop driving video', flush=True)
        driving_video_path = process_video(
            driving_origin_path, '/tmp/driving.mp4', min_frames=15, face_data=face_data)
        torch.cuda.empty_cache()
    else:
        driving_video_path = driving_origin_path
    command = f"ffmpeg -y -i {driving_video_path} /tmp/temp.wav "
    subprocess.call(command, shell=True)
    # driving_video_path = video_gpen_process(
    #     driving_video_path, model_dir, out_video_path='/tmp/driving_enhace.mp4')
    print('create animation', flush=True)
    safa_model_path = f'{model_dir}/final_3DV.tar'
    safa_video = create_image_animation(source_path, driving_video_path, '/tmp/temp.mp4', config_path,
                                        safa_model_path, with_eye=True, relative=True, adapt_scale=True, use_best_frame=use_best)
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
    # -preset veryslow
    command = f"ffmpeg -y -i {paste_video_path} -i /tmp/temp.wav  -crf  {crf} -vcodec h264  {result_path} "
    subprocess.call(command, shell=True)


if __name__ == '__main__':

    # root='/home/yuan/hdd/safa_test/01_18_2'
    # make_image_animation_dataflow(f'{root}/EP010-08.jpg',f'{root}/1.mp4',f'{root}/1_gfpgan.mp4','ckpt/final_3DV.tar',use_crop=False)
    # concat_video(f'{root}/1_gfpgan.mp4',f'{root}/out/1.mp4','concat2.mp4')
    from mock import generate_lip_video
    # root = '/home/yuan/hdd/05_25'
    # driving_video_path = os.path.join(
    #     root, 'lip/result_woman.mp4')
    # face_data = '/home/yuan/hdd/driving_video/model2/face.pkl'
    from pathlib import Path
    from glob import glob

    # for image_path in sorted(glob(f'{root}/img/*g')):
    #     # image_input = os.path.join(root, image_path)
    #     image_input = image_path
    #     save_dir = os.path.join(root, Path(
    #         image_path).parent.name+'_out_modify')
    #     os.makedirs(save_dir, exist_ok=True)

    #     out_path = os.path.join(save_dir, 'result_' +
    #                             Path(image_input).stem+'.mp4')
    #     if os.path.exists(out_path):
    #         continue
    #     make_image_animation_dataflow(
    #         image_input, driving_video_path, out_path, 'ckpt/', use_crop=True, face_data=face_data)
    #     break
    # root = '/home/yuan/hdd/05_19'
    # for audio_path in sorted(glob(f'{root}/audio/*wav')):
    #     image_input = f"{root}/img/412.png"
    #     lip_dir = f'{root}/lip'
    #     os.makedirs(lip_dir, exist_ok=True)
    #     lip_path = os.path.join(lip_dir,
    #                             Path(audio_path).stem+'.mp4')
    #     if os.path.exists(lip_path) == False:
    #         generate_lip_video(
    #             "datadir/preprocess/driving_woman/face.pkl", audio_path, lip_path)
    #     save_dir = os.path.join(root, Path(
    #         image_input).parent.name+'_out')
    #     os.makedirs(save_dir, exist_ok=True)
    #     out_path = os.path.join(save_dir, 'result_' +
    #                             Path(lip_path).stem+'.mp4')
    #     if os.path.exists(out_path):
    #         continue
    #     make_image_animation_dataflow(
    #         image_input, lip_path, out_path, 'ckpt/', use_crop=True, crf=10, face_data="datadir/preprocess/driving_woman/face.pkl", use_best=True)

    root = '/home/yuan/hdd/06_01'
    # face_data = "datadir/preprocess/driving_woman/face.pkl"
    face_data = '/home/yuan/hdd/driving_video/model2/face.pkl'
    for audio_path in sorted(glob(f'{root}/audio/*')):
        image_input = f"{root}/img/0429_1-ok.png"
        lip_dir = f'{root}/lip'
        os.makedirs(lip_dir, exist_ok=True)
        lip_path = os.path.join(lip_dir, Path(audio_path).stem+'.mp4')
        if os.path.exists(lip_path) == False:
            generate_lip_video(
                face_data, audio_path, lip_path)
        save_dir = os.path.join(root, Path(
            image_input).parent.name+'_out')
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, 'result_' +
                                Path(audio_path).stem+'.mp4')
        if os.path.exists(out_path):
            continue
        make_image_animation_dataflow(
            image_input, lip_path, out_path, 'ckpt/', use_crop=True, face_data=face_data)

    # file_list = os.listdir('/home/yuan/hdd/05_25/img_out')
    # l_dir = '/home/yuan/hdd/05_25/img_out/'
    # r_dir = '/home/yuan/hdd/05_25/img_out_tps/'
    # o_dir = '/home/yuan/hdd/05_25/concat/'
    # for file in file_list:
    #     l = l_dir+file
    #     r = r_dir+file
    #     o = o_dir+file
    #     if os.path.exists(o):
    #         continue
    #     concat_video(l, r, o)
