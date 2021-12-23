import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
from imageio import mimread




def extract_imgs_from_video(video_name):
    out_video_dir = os.path.join(png_dir, video_name)
    if not os.path.exists(out_video_dir):
        os.mkdir(out_video_dir)
    video_pth = os.path.join(video_dir, video_name)
    video = np.array(mimread(video_pth,memtest=False))

    for i in range(video.shape[0]):
        if i % 5 == 0:
            cv2.imwrite(os.path.join(out_video_dir, str(i) + '.png'), video[i, :, :, ::-1])


if __name__ == "__main__":
    video_dir = '/home/yuan/hdd/video'
    png_dir = '/home/yuan/hdd/png'
    if not os.path.exists(png_dir):
        os.mkdir(png_dir)
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=(cores // 2))
    for _ in tqdm(pool.imap_unordered(extract_imgs_from_video, os.listdir(video_dir))):
        pass
