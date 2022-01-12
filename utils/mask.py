import cv2
import numpy as np

def blur_image(image, kernel_size=None):
    if kernel_size is None:
        kernel_size = np.random.randint(5, 15)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    kernel = np.ones(kernel_size,
                     dtype=np.float32) / np.prod(kernel_size, dtype=np.float32)
    blur_img = cv2.filter2D(image, ddepth=cv2.CV_8U, dst=-1, kernel=kernel,
                            anchor=(-1, -1), delta=0, borderType=cv2.BORDER_REPLICATE)
    return blur_img

def _cal_mouth_contour_mask(landmarks, img_height, img_width,
                            mask_bound=None,
                            shrink_width_ratio=0.1, expand_height_ratio=0.1):
    """In cv2 format (H, W, C)"""
    # mouth_landmarks = landmarks[48:]
    left_face_width = landmarks[33, 0] - landmarks[2, 0]
    right_face_width = landmarks[14, 0] - landmarks[33, 0]

    delta_left_face_width = left_face_width * shrink_width_ratio
    delta_right_face_width = right_face_width * shrink_width_ratio

    delta_face_height = (landmarks[8, 1] -
                         landmarks[33, 1]) * expand_height_ratio
    mouth_contours = [[
        [landmarks[2, 0] + delta_left_face_width, landmarks[2, 1]],
        [landmarks[4, 0] + delta_left_face_width, landmarks[4, 1]],
        [landmarks[7, 0], landmarks[7, 1] + delta_face_height],
        [landmarks[8, 0], landmarks[8, 1] + delta_face_height],
        [landmarks[9, 0], landmarks[9, 1] + delta_face_height],
        [landmarks[12, 0] - delta_right_face_width, landmarks[12, 1]],
        [landmarks[14, 0] - delta_right_face_width, landmarks[14, 1]],
        [landmarks[28, 0], landmarks[28, 1] + delta_face_height],
    ]]
    mouth_contours = np.array(mouth_contours, dtype=np.int32)
    max_val = landmarks[28, 1] + delta_face_height
    mouth_contours[:, :, 1] = np.maximum(mouth_contours[:, :, 1], max_val)

    # LIMIT BORDER
    if mask_bound is not None:
        crop_x0, crop_x1, crop_y0, crop_y1 = mask_bound
        mouth_contours[:, :, 1] = np.maximum(mouth_contours[:, :, 1], crop_y0)
        mouth_contours[:, :, 1] = np.minimum(mouth_contours[:, :, 1], crop_y1)
        mouth_contours[:, :, 0] = np.maximum(mouth_contours[:, :, 0], crop_x0)
        mouth_contours[:, :, 0] = np.minimum(mouth_contours[:, :, 0], crop_x1)

    mask = cv2.drawContours(
        np.ones((img_height, img_width, 1), dtype=np.uint8) * 255, mouth_contours, -1, (0, 0, 0), -1)
    mask = blur_image(mask, 29)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = np.expand_dims(mask,-1)
    # mask=255-mask
    mask = mask.astype(np.float32) / 255.0
    return mask
