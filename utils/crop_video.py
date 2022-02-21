import subprocess
import face_alignment
# import skimage.io
# import numpy
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm
import os
import imageio
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def extract_bbox(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) == 0:
        return []
    return np.array(bboxes)[:, :-1] * scale_factor



def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)


def compute_bbox(start, end, fps, tube_bbox, frame_shape, inp, image_shape,output, increase_area=0.1):
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top

    #Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    h, w = bot - top, right - left

    start = start / fps
    end = end / fps
    time = end - start

    scale = f'{image_shape[0]}:{image_shape[1]}'

    # return f'ffmpeg -i {inp} -ss {start} -t {time} -filter:v "crop={w}:{h}:{left}:{top}, scale={scale}" crop.mp4'
    return f'ffmpeg  -y -i {inp}  -filter:v "crop={w}:{h}:{left}:{top}, scale={scale}" {output}'


def compute_bbox_trajectories(trajectories, fps, frame_shape, inp,image_shape,min_frames,increase,output):
    commands = []
    for i, (bbox, tube_bbox, start, end) in enumerate(trajectories):
        if (end - start) > min_frames:
            command = compute_bbox(start, end, fps, tube_bbox, frame_shape, inp=inp, image_shape=image_shape, increase_area=increase,output=output)
            commands.append(command)
    return commands


def process_video(inp,output,image_shape=(256,256),increase=0.1,iou_with_initial=0.25,min_frames=150,device='cuda'):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    video = imageio.get_reader(inp)

    trajectories = []
    previous_frame = None
    fps = video.get_meta_data()['fps']
    commands = []
    try:
        for i, frame in enumerate(tqdm(video)):
            frame_shape = frame.shape
            bboxes =  extract_bbox(frame, fa)
            ## For each trajectory check the criterion
            not_valid_trajectories = []
            valid_trajectories = []

            for trajectory in trajectories:
                tube_bbox = trajectory[0]
                intersection = 0
                for bbox in bboxes:
                    intersection = max(intersection, bb_intersection_over_union(tube_bbox, bbox)) #staartframe bbox, now frame bbox
                if intersection > iou_with_initial:
                    valid_trajectories.append(trajectory)
                else:
                    not_valid_trajectories.append(trajectory)
            # print("frame",i,"not trajectories",not_valid_trajectories)
            commands += compute_bbox_trajectories(not_valid_trajectories, fps, frame_shape,inp,image_shape,min_frames,increase,output)# return none
            trajectories = valid_trajectories

            ## Assign bbox to trajectories, create new trajectories
            for bbox in bboxes:
                intersection = 0
                current_trajectory = None
                for trajectory in trajectories:
                    tube_bbox = trajectory[0]
                    current_intersection = bb_intersection_over_union(tube_bbox, bbox)
                    if intersection < current_intersection and current_intersection > iou_with_initial:
                        intersection = bb_intersection_over_union(tube_bbox, bbox)
                        current_trajectory = trajectory

                ## Create new trajectory
                if current_trajectory is None:
                    trajectories.append([bbox, bbox, i, i]) #bbox ,tube_bbox,start,end   : tube_bbox is max bbox
                else:
                    current_trajectory[3] = i
                    current_trajectory[1] = join(current_trajectory[1], bbox)
            # print("frame",i,"trajecties",trajectories)

    except IndexError as e:
        raise (e)

    commands += compute_bbox_trajectories(trajectories, fps, frame_shape, inp,image_shape,min_frames,increase,output)
    command=commands[0]
    subprocess.call(command,shell=True)
    return output
    # return commands


if __name__ == "__main__":
    inp='/home/yuan/repo/my_safa/mock_dir/driving_man.mp4'
    output='/home/yuan/repo/Talking-Face_PC-AVS/misc/test_driving_increase-01.mp4'
    commands = process_video(inp,output,image_shape=(224,224),increase=-0.1)
    # for command in commands:
    #     print (command)

        