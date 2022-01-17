import pickle
import cv2
import numpy as np
import os
import torch
from animation_demo import create_video_animation
from utils.face_restore_helper import create_face_helper
from utils.mask import _cal_mouth_contour_mask
from utils.blend import LaplacianBlending
import face_alignment
import subprocess
from tqdm import trange

def concat_video(left,right,out_path):
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
        frame = np.concatenate([frame1,frame2], axis=1)
        out.write(frame)
    video1.release()
    video2.release()
    out.release()

def check(dir):
    source_video=cv2.VideoCapture(f'{dir}/source.mp4')
    result_video=cv2.VideoCapture(f'{dir}/driving.mp4')
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
        frame=frame.astype(np.uint8)
        out.write(frame)
    out.release()
    source_video.release()
    result_video.release()

def extract_landmark(video_path, out_path):

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)
    video = cv2.VideoCapture(video_path)
    all_landmarks = []
    frame_count=int(video.get(7))
    for _ in  trange(frame_count):
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

def face_preprocess(input_video,landmark_file,save_name):
    face_helper = create_face_helper(512)
    video=cv2.VideoCapture(input_video)
    f=open(landmark_file,'rb')
    video_landmarks = pickle.load(f)
    out_path=f'{save_name}.mp4'
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(
        *'MP4V'), 30.0, (256,256))
    matrix_list=[]
    new_landmark_list=[]
    data_len=min(len(video_landmarks),int(video.get(7)))
    for i in trange(data_len):
        _,frame=video.read()
        frame_landmark=video_landmarks[i]
        if frame_landmark is None:
            matrix_list.append(None)
            new_landmark_list.append(None)
            large_face=np.zeros((256,256,3),np.uint8)
        else:
            face_helper.clean_all()
            face_helper.read_image(frame)
            face_helper.get_face_landmarks_3(frame_landmark, only_keep_largest=True, eye_dist_threshold=5)
            face_helper.align_warp_face()

            assert len(face_helper.cropped_faces) != 0
            

            affine_matrix = face_helper.affine_matrices[0]
            large_face = face_helper.cropped_faces[0]
            affine_landmarks = (np.matmul(affine_matrix[:, :2], frame_landmark.T) + affine_matrix[:, 2:]).T
            matrix_list.append(affine_matrix)
            new_landmark_list.append(affine_landmarks)
        out.write(cv2.resize(large_face,(256,256)))
    out_data_path=f'{save_name}_data.pkl'
    with open(out_data_path,'wb') as f:
            pickle.dump({'matrix':matrix_list,'affine_landmark':new_landmark_list},f)
    return out_path,out_data_path



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
def paste_origin_video(source_origin_path,safa_video_path,temp_dir,landmark_path,source_video_data_path):
    full_video=cv2.VideoCapture(source_origin_path)
    crop_video=cv2.VideoCapture(safa_video_path)
    out_video_path=f'{temp_dir}/paste_temp.mp4'
    h= int(full_video.get(4))
    w= int(full_video.get(3))
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(
        *'XVID'), 30.0, (w,h))
    lb=create_lb(4)
    with open(landmark_path,'rb' ) as f:
        source_landmark=pickle.load(f)
    with open(source_video_data_path,'rb') as f:
        source_data= pickle.load(f)
        soucre_matrix=source_data['matrix']
        # source_affine_landmark=source_data['affine_landmark'] #512*512  landmark
    frame_count= int(crop_video.get(7))#min(len(source_affine_landmark),len(drive_data_affine_landmark))
    for i in  trange(frame_count):
        _,full_frame=full_video.read()
        _,crop_frame=crop_video.read()

        if source_landmark[i] is None:
            full_out= full_frame.copy()
        else:
            crop_frame=cv2.resize(crop_frame,(512,512))
            inv_matrix=cv2.invertAffineTransform(soucre_matrix[i])
            new_frame = cv2.warpAffine(
                    crop_frame, inv_matrix,(w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132)) 
            mask=_cal_mouth_contour_mask(source_landmark[i],h,w,None,0.1)

            x1,x2,y1,y2=quantize_position(0,w,0,h,4)
            if x2>w or y2>h:
                mask= cv2.copyMakeBorder(mask,0,y2-h,0,x2-w,cv2.BORDER_CONSTANT,value=[255])
                mask=mask.reshape(y2,x2,1)
                full_frame=cv2.copyMakeBorder(full_frame,0,y2-h,0,x2-w,cv2.BORDER_CONSTANT,value=[255,255,255])
                new_frame=cv2.copyMakeBorder(new_frame,0,y2-h,0,x2-w,cv2.BORDER_CONSTANT,value=[255,255,255])
            mask_tesor=torch.tensor(mask,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
            y_tensor=torch.tensor(full_frame/255,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
            x_tensor=torch.tensor(new_frame/255,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
            out=lb(y_tensor,x_tensor,mask_tesor)
            full_out=(out[0][:,:h,:w].permute(1,2,0)*255).numpy().astype(np.uint8)
        out_video.write(full_out)
    out_video.release()
    # out_video2.release()
    full_video.release()
    crop_video.release()
    return out_video_path



    

def make_animation_dataflow(source_origin_path,driving_origin_path,temp_dir,result_path,model_path,config_path=None,add_audo=False):
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
        config_path=f"{os.path.split(os.path.realpath(__file__))[0]}/config/end2end.yaml"
    print('extract_landmark: source')
    landmark_path=extract_landmark(source_origin_path,f'{temp_dir}/source.pkl')
    print('crop_video: source')
    source_video_path,source_data_path=face_preprocess(source_origin_path,landmark_path,f'{temp_dir}/source')
    print('crop_video: driving')
    driving_video_path,driving_video_data_path=face_preprocess(driving_origin_path,landmark_path,f'{temp_dir}/driving')
    print('generate safa_result_video')
    safa_video_path=create_video_animation(source_video_path,driving_video_path,f'{temp_dir}/temp.mp4',config_path,model_path,with_eye=True,relative=False,adapt_scale=True)
    print('paste safa_result_video on the origin source video ')
    temp_paste_video_path=paste_origin_video(source_origin_path,safa_video_path,temp_dir,landmark_path,source_data_path)
    video=cv2.VideoCapture(temp_paste_video_path)
    fps=video.get(5)
    video.release()
    if add_audo:
        command=f"ffmpeg -y -i {driving_origin_path} {temp_dir}/temp.wav "
        subprocess.call(command,shell=True)
        command=f"ffmpeg -y -i {temp_paste_video_path} -i {temp_dir}/temp.wav -vf fps={fps} -crf 0 -vcodec h264  {result_path} " #-preset veryslow
        subprocess.call(command,shell=True)
    else:
        command=f"ffmpeg -y -i {temp_paste_video_path}  -vf fps={fps} -crf 0 -vcodec h264  {result_path} " #-preset veryslow
        subprocess.call(command,shell=True)
if __name__ == '__main__':
    # inference_animation_dataflow('new_test/source_all.mp4','new_test/driving_all.mp4','temp','finish.mp4','ckpt/final_3DV.tar')
    make_animation_dataflow('finish_1/finish.mp4','finish_1/driving_all.mp4','finish_1/temp2','finish_t.mp4','ckpt/final_3DV.tar',add_audo=True)
    # make_animation_dataflow('finish.mp4','finish_2/driving_all.mp4','finish_2/temp','finish2.mp4','ckpt/final_3DV.tar',add_audo=True)





# ffmpeg -i test/input1.mp4  -filter:v "crop=476:476:733:151, scale=256:256" crop.mp4
# x 733:1209
# y 151:627