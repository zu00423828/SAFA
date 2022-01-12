import pickle
import cv2
import numpy as np
import os
import torch
from animation_demo import create_video_animation
from utils.face_restore_helper import create_face_helper
from utils.mask import _cal_mouth_contour_mask
from utils.blend import LaplacianBlending
import subprocess
from tqdm import trange
def cross_point(line1,line2):#計算交點函數
    x1=line1[0][0]#取四點座標
    y1=line1[0][1]
    x2=line1[1][0]
    y2=line1[1][1]
    
    x3=line2[0][0]
    y3=line2[0][1]
    x4=line2[1][0]
    y4=line2[1][1]
    
    k1=(y2-y1)*1.0/(x2-x1)#計算k1,由於點均爲整數，需要進行浮點數轉化
    b1=y1*1.0-x1*k1*1.0#整型轉浮點型是關鍵
    if (x4-x3)==0:#L2直線斜率不存在操作
        k2=None
        b2=0
    else:
        k2=(y4-y3)*1.0/(x4-x3)#斜率存在操作
        b2=y3*1.0-x3*k2*1.0
    if k2==None:
        x=x3
    else:
        x=(b2-b1)*1.0/(k1-k2)
    y=k1*x*1.0+b1*1.0
    return [int(x),int(y)]
def concat_video(dir):
    video1 = cv2.VideoCapture(f'{dir}/source.mp4')
    video2 = cv2.VideoCapture(f'{dir}/driving.mp4')
    video3 = cv2.VideoCapture(f'{dir}/result.mp4')

    out = cv2.VideoWriter(
        f'{dir}/concat.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (768, 256))
    while video1.isOpened():
        ret, frame1 = video1.read()
        if not ret:
            break
        ret, frame2 = video2.read()
        if not ret:
            break
        ret, frame3 = video3.read()
        if not ret:
            break
        frame = np.concatenate([frame1,frame2,frame3], axis=1)
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
    import face_alignment
    import pickle
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)
    video = cv2.VideoCapture(video_path)
    all_landmarks = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        result = fa.get_landmarks(frame)
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
    print(len(video_landmarks),video.get(7))
    for i in trange(len(video_landmarks)):
        _,frame=video.read()
        frame_landmark=video_landmarks[i]
        face_helper.clean_all()
        face_helper.read_image(frame)
        face_helper.get_face_landmarks_3(frame_landmark, only_keep_largest=True, eye_dist_threshold=5)
        face_helper.align_warp_face()

        assert len(face_helper.cropped_faces) != 0
        

        affine_matrix = face_helper.affine_matrices[0]
        large_face = face_helper.cropped_faces[0]
        # face = face_helper.cropped_small_faces[0]
        affine_landmarks = (np.matmul(affine_matrix[:, :2], frame_landmark.T) + affine_matrix[:, 2:]).T
        matrix_list.append(affine_matrix)
        new_landmark_list.append(affine_landmarks)
        # cv2.imwrite(f'frame/{i}.png',large_face)
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


def paste_origin_video(source_origin_path,safa_video_path,temp_dir,landmark_path,source_video_data_path):
    full_video=cv2.VideoCapture(source_origin_path)
    crop_video=cv2.VideoCapture(safa_video_path)
    out_video_path=f'{temp_dir}/paste_temp.mp4'
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(
        *'XVID'), 30.0, (1920,1080))
    lb=create_lb(3)
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
        crop_frame=cv2.resize(crop_frame,(512,512))
        inv_matrix=cv2.invertAffineTransform(soucre_matrix[i])
        new_frame = cv2.warpAffine(
                crop_frame, inv_matrix,(1920,1080), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132)) 
        mask=_cal_mouth_contour_mask(source_landmark[i],1080,1920,None,0.1)
        mask_tesor=torch.tensor(mask,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
        y_tensor=torch.tensor(full_frame/255,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
        x_tensor=torch.tensor(new_frame/255,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
        out=lb(y_tensor,x_tensor,mask_tesor)
        full_out=(out[0].permute(1,2,0)*255).numpy().astype(np.uint8)
        out_video.write(full_out)
    out_video.release()
    # out_video2.release()
    full_video.release()
    crop_video.release()
    return out_video_path



    

def inference_animation_dataflow(source_origin_path,driving_origin_path,temp_dir,result_path,model_path,config_path=None,add_audo=False):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    print('extract_landmark: source')
    landmark_path=extract_landmark(source_origin_path,f'{temp_dir}/source.pkl')
    # landmark_path='temp/source.pkl'
    print('crop_video: source')
    source_video_path,source_data_path=face_preprocess(source_origin_path,landmark_path,f'{temp_dir}/source')
    print('crop_video: driving')
    driving_video_path,driving_video_data_path=face_preprocess(driving_origin_path,landmark_path,f'{temp_dir}/driving')
    print('generate safa_result_video')
    safa_video_path=create_video_animation(source_video_path,driving_video_path,f'{temp_dir}/temp.mp4',config_path,model_path,with_eye=True,relative=False,adapt_scale=True)
    print('paste safa_result_video on the origin source video ')
    if config_path is None:
        config_path=f"{os.path.split(os.path.realpath(__file__))[0]}/config/end2end.yaml"
    print(config_path)
    temp_paste_video_path=paste_origin_video(source_origin_path,safa_video_path,temp_dir,landmark_path,source_data_path)
    print(temp_paste_video_path)
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
    # inference_animation_dataflow('new_test/source_all.mp4','new_test/driving_all.mp4','temp','finish.mp4','config/end2end.yaml','ckpt/final_3DV.tar')
    inference_animation_dataflow('finish.mp4','finish_1/driving_all.mp4','finish_1/temp','finish1.mp4','ckpt/final_3DV.tar',config_path='config/end2end.yaml',add_audo=True)
    inference_animation_dataflow('finish.mp4','finish_2/driving_all.mp4','finish_2/temp','finish2.mp4','ckpt/final_3DV.tar',config_path='config/end2end.yaml',add_audo=True)





# ffmpeg -i test/input1.mp4  -filter:v "crop=476:476:733:151, scale=256:256" crop.mp4
# x 733:1209
# y 151:627