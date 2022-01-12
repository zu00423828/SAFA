from facexlib.utils.face_restoration_helper import FaceRestoreHelper, imwrite, get_largest_face, get_center_face
import numpy as np
import cv2
import os
from io import BytesIO
class Landmarks68:

    def __init__(self, landmarks, width=None, height=None, verbose=False):
        assert isinstance(landmarks, np.ndarray)
        self.is_batch = len(landmarks.shape) == 3
        self._landmarks = landmarks
        self.width = width
        self.height = height
        self.reduce_axis = 1 if self.is_batch else 0
        self.is_normed = self._landmarks.mean() < 1.0
        self.verbose = verbose

    def _need_border(self):
        assert self.width is not None
        assert self.height is not None

    def norm(self):
        if self.is_normed:
            return
        self._need_border()
        self._landmarks = self._landmarks.astype(np.float32)
        self._landmarks[..., 0] = self._landmarks[..., 0] / self.width
        self._landmarks[..., 1] = self._landmarks[..., 1] / self.height
        self.is_normed = True

    def unnorm(self):
        if not self.is_normed:
            return
        self._need_border()
        self._landmarks[..., 0] = self._landmarks[..., 0] * self.width
        self._landmarks[..., 1] = self._landmarks[..., 1] * self.height
        self.is_normed = False

    def limit_border(self):
        if self.is_normed:
            self._landmarks[..., 0] = np.minimum(1.0, np.maximum(0, self._landmarks[..., 0]))
            self._landmarks[..., 1] = np.minimum(1.0, np.maximum(0, self._landmarks[..., 1]))
        else:
            self._need_border()
            self._landmarks[..., 0] = np.minimum(self.width, np.maximum(0, self._landmarks[..., 0]))
            self._landmarks[..., 1] = np.minimum(self.height, np.maximum(0, self._landmarks[..., 1]))

    def get_left_eyes(self, reduce_func=None):
        return self.get_points(range(36, 42), reduce_func=reduce_func)

    def get_right_eyes(self, reduce_func=None):
        return self.get_points(range(42, 48), reduce_func=reduce_func)

    def get_eyes(self, reduce_func=None):
        return self.get_points(range(36, 48), reduce_func=reduce_func)

    def get_nose(self, reduce_func=None):
        return self.get_points([33], reduce_func=np.sum)

    def get_eyebrow(self, reduce_func=None):
        return self.get_points([27], reduce_func=np.sum)

    def get_mouths(self, reduce_func=None):
        return self.get_points([51, 57, 62, 66], reduce_func=reduce_func)

    def get_chin(self, reduce_func=None):
        return self.get_points([8], reduce_func=np.sum)
        # return self.get_points(range(7, 11), reduce_func=reduce_func)

    def get_face_contours(self):
        return self.get_points(range(17))

    def get_points(self, indices, reduce_func=None):
        if self.is_batch:
            arr = self._landmarks[:, indices]
        else:
            arr = self._landmarks[indices]
        if reduce_func is not None:
            arr = reduce_func(arr, axis=self.reduce_axis)
        return arr

    def distances(self):
        assert self.is_batch
        diffs = np.diff(self._landmarks, axis=0)
        dists = (diffs.astype(np.float32) ** 2).sum(1).sum(1)
        return dists

    def stat(self):
        assert self.is_batch, "stat only available for batch data"
        methods = {
            'left_eye': self.get_left_eyes,
            'right_eye': self.get_right_eyes,
            'nose': self.get_nose,
            'mouths': self.get_mouths,
            'chin': self.get_chin,
        }

        row = {}
        for key, method in methods.items():
            arr = method(np.mean)
            row[f"{key}_min_x"] = arr[:, 0].min()
            row[f"{key}_min_y"] = arr[:, 1].min()
            row[f"{key}_max_x"] = arr[:, 0].max()
            row[f"{key}_max_y"] = arr[:, 1].max()
            row[f"{key}_mean_x"] = arr[:, 0].mean()
            row[f"{key}_mean_y"] = arr[:, 1].mean()
            row[f"{key}_std_x"] = arr[:, 0].std()
            row[f"{key}_std_y"] = arr[:, 1].std()
        return row

    def is_valid(self):
        assert self.is_normed, "you should norm landmarks before is_valid"
        if self.is_batch:
            min_dist = self.distances().min()
            if min_dist > 0.1:
                if self.verbose:
                    print("invalid dists:", min_dist)
                return False

        eye_y = self.get_eyes(np.mean)[..., 1]
        max_eye_y = eye_y.max()
        if max_eye_y > 0.320:
            if self.verbose:
                print("bad max_eye_y", max_eye_y)
            return False
        min_eye_y = eye_y.min()
        if min_eye_y < 0.0177:
            if self.verbose:
                print("bad min_eye_y", min_eye_y)
            return False

        # **** NOSE IS ALWAYS CENTERED ****
        # nose = self.get_nose()
        # min_nose_y = nose[..., 1].min()
        # if min_nose_y < 0.218:
        #     if self.verbose:
        #         print("bad min_nose_y", min_nose_y)
        #     return False
        # max_nose_y = nose[..., 1].max()
        # if max_nose_y > 0.6581:
        #     if self.verbose:
        #         print("bad max_nose_y", max_nose_y)
        #     return False
        # min_nose_x = nose[..., 0].min()
        # if min_nose_x < 0.265:
        #     if self.verbose:
        #         print("bad min_nose_x", min_nose_x)
        #     return False
        # max_nose_x = nose[..., 0].max()
        # if max_nose_x > 0.7285:
        #     if self.verbose:
        #         print("bad max_nose_x", max_nose_x)
        #     return False
        # *********************************

        mouths = self.get_mouths(np.mean)
        min_mouth_x = mouths[..., 0].min()
        if min_mouth_x < 0.444:
            if self.verbose:
                print("bad min_mouth_x", min_mouth_x)
            return False
        max_mouth_x = mouths[..., 0].max()
        if max_mouth_x > 0.559:
            if self.verbose:
                print("bad max_mouth_x", max_mouth_x)
            return False
        min_mouth_y = mouths[..., 1].min()
        if min_mouth_y < 0.480:
            if self.verbose:
                print("bad min_mouth_y", min_mouth_y)
            return False
        max_mouth_y = mouths[..., 1].max()
        if max_mouth_y > 0.663:
            if self.verbose:
                print("bad max_mouth_y", max_mouth_y)
            return False
        return True

    @classmethod
    def from_npy(cls, path):
        landmarks = np.load(path)
        return cls(landmarks)

    @classmethod
    def from_bytes(cls, bytes):
        buf = BytesIO(bytes)
        return cls.from_npy(buf)

class LandmarksFaceRestoreHelper(FaceRestoreHelper):


    def __init__(self,
                 upscale_factor,
                 face_size=512,
                 crop_ratio=(1, 1),
                 save_ext='png',
                 pad_blur=False,
                 current_smooth_weight=0.6):
        self.template_3points = True  # improve robustness
        self.upscale_factor = upscale_factor
        # the cropped face ratio based on the square face
        self.crop_ratio = crop_ratio  # (h, w)
        assert (self.crop_ratio[0] >= 1 and self.crop_ratio[1] >= 1), 'crop ration only supports >=1'
        self.face_size = (int(face_size * self.crop_ratio[1]), int(face_size * self.crop_ratio[0]))

        if self.template_3points:
            self.face_template = np.array([
                [256-64, 240],
                [256+64, 240],
                [256, 314.01935],
                [256, 388],
                [256, 476.8],
        ])
        else:
            # standard 5 landmarks for FFHQ faces with 512 x 512
            self.face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                           [201.26117, 371.41043], [313.08905, 371.15118]])

        self.face_template = self.face_template * (face_size / 512.0)
        if self.crop_ratio[0] > 1:
            self.face_template[:, 1] += face_size * (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1:
            self.face_template[:, 0] += face_size * (self.crop_ratio[1] - 1) / 2
        self.save_ext = save_ext
        self.pad_blur = pad_blur
        if self.pad_blur is True:
            self.template_3points = False

        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.cropped_small_faces = []
        self.restored_faces = []
        self.pad_input_imgs = []

        self._last_affine_matrix = None
        assert 0.0 < current_smooth_weight <= 1.0
        self._prev_ratio = 1.0 - current_smooth_weight
        self._current_ratio = current_smooth_weight

        self.crop_y0 = int(self.face_size[1] * 160 / 512)
        self.crop_y1 = self.face_size[1]
        self.crop_x0 = int(self.face_size[0] * 80 / 512)
        self.crop_x1 = int(self.face_size[0] * 432 / 512)
        self.crop_width = self.crop_x1 - self.crop_x0
        self.crop_height = self.crop_y1 - self.crop_y0

    def set_face_template(self, left_eye, right_eye, eyebrow):
        bleye = eyebrow - left_eye
        rbeye = right_eye - eyebrow
        half_d = (np.sqrt(rbeye[0] ** 2 + rbeye[1] ** 2) + np.sqrt(bleye[0] ** 2 + bleye[1] ** 2)) / 2.0
        dbl = np.sqrt(bleye[0] ** 2 + bleye[1] ** 2)
        drb = np.sqrt(rbeye[0] ** 2 + rbeye[1] ** 2)
        left_width = max(0, 64 * dbl / half_d)
        right_width = max(0, 64 * drb / half_d)
        self.face_template = np.array([
            [256-left_width, 240],
            [256+right_width, 240],
            [256, 314.01935],
            [256, 388],
            [256, 476.8],
        ])

    def reset_smooth(self):
        self._last_landmark = None

    def align_warp_face(self, save_cropped_path=None, border_mode='constant'):
        """Align and warp faces with face template.
        """
        if self.pad_blur:
            assert len(self.pad_input_imgs) == len(
                self.all_landmarks_5), f'Mismatched samples: {len(self.pad_input_imgs)} and {len(self.all_landmarks_5)}'
        for idx, landmark in enumerate(self.all_landmarks_5):
            # use 5 landmarks to get affine matrix
            affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template)[0]

            # # **** horizonal eyes ****
            # eyes = landmark[:2].T
            # affine_eyes = (np.matmul(affine_matrix[:, :2], eyes) + affine_matrix[:, 2:])
            # lx, ly = affine_eyes[0, 0], affine_eyes[1, 0]
            # rx, ry = affine_eyes[0, 1], affine_eyes[1, 1]
            # dth = np.math.atan2(-(ry - ly), rx - lx)
            # dR = np.array([[np.cos(dth), np.sin(dth)], [-np.sin(dth), np.cos(dth)]])
            # affine_matrix[:, :2] = np.matmul(dR, affine_matrix[:, :2])
            # # **************************

            # **** smooth matrix ****
            if self._last_affine_matrix is not None:
                affine_matrix = self._last_affine_matrix * self._prev_ratio + affine_matrix * self._current_ratio
            # ***********************

            # **** force center nose ******
            nose = landmark[2:3].T
            affine_nose = (np.matmul(affine_matrix[:, :2], nose) + affine_matrix[:, 2:])
            delta_nose = np.array([[256 - affine_nose[0, 0]], [314.01935 - affine_nose[1, 0]]])
            affine_matrix[:, 2:] = affine_matrix[:, 2:] + delta_nose
            # *********************

            self.affine_matrices.append(affine_matrix)
            self._last_affine_matrix = affine_matrix
            # warp and crop faces
            if border_mode == 'constant':
                border_mode = cv2.BORDER_CONSTANT
            elif border_mode == 'reflect101':
                border_mode = cv2.BORDER_REFLECT101
            elif border_mode == 'reflect':
                border_mode = cv2.BORDER_REFLECT
            if self.pad_blur:
                input_img = self.pad_input_imgs[idx]
            else:
                input_img = self.input_img
            cropped_face = cv2.warpAffine(
                input_img, affine_matrix, self.face_size, borderMode=border_mode, borderValue=(135, 133, 132))  # gray
            self.cropped_faces.append(cropped_face)
            self.cropped_small_faces.append(cropped_face[self.crop_y0:self.crop_y1, self.crop_x0:self.crop_x1])
            # save the cropped face
            if save_cropped_path is not None:
                path = os.path.splitext(save_cropped_path)[0]
                save_path = f'{path}_{idx:02d}.{self.save_ext}'
                imwrite(cropped_face, save_path)

    def get_face_landmarks_3(self,
                             landmarks_68_in_image,
                             only_keep_largest=False,
                             only_center_face=False,
                             resize=None,
                             blur_ratio=0.01,
                             eye_dist_threshold=None):
        # if resize is None:
        #     scale = 1
        #     input_img = self.input_img
        # else:
        #     h, w = self.input_img.shape[0:2]
        #     scale = min(h, w) / resize
        #     h, w = int(h / scale), int(w / scale)
        #     input_img = cv2.resize(self.input_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        l = Landmarks68(landmarks_68_in_image)
        left_eye = l.get_left_eyes(np.mean)
        right_eye = l.get_right_eyes(np.mean)
        nose = l.get_nose(np.mean)
        mouth_center = l.get_mouths(np.mean)
        eyebrow = l.get_eyebrow(np.mean)
        chin = l.get_chin(np.mean)

        self.set_face_template(left_eye, right_eye, eyebrow)
        # left_mouth = landmarks_68_in_image[48]
        # right_mouth = landmarks_68_in_image[54]
        # mid_mouth = (landmarks_68_in_image[51] + landmarks_68_in_image[57]) / 2.0
        # left_mouth = landmarks_68_in_image[48]
        # right_mouth = landmarks_68_in_image[54]
        # m = right_mouth - left_mouth
        # left_mouth = m * 0.3 + left_mouth
        # right_mouth = m * 0.7 + left_mouth
        # eyebrow = Landmarks68.cal_eyebrow(left_eye, right_eye, nose)
        bboxes = [np.array([0.0, 0.0, 0.0, 0.0, 0.0,
            left_eye[0], left_eye[1],
            right_eye[0], right_eye[1],
            nose[0], nose[1],
            mouth_center[0], mouth_center[1],
            chin[0], chin[1],
            # 0.0, 0.0
        ])]
        # with torch.no_grad():
        #     bboxes = self.face_det.detect_faces(input_img, 0.97) * scale
        #     if len(bboxes) == 0:
        #         print("re detect")
        #         bboxes = self.large_helper.face_det.detect_faces(input_img, 0.97) * scale
        #         print("got bboxes", len(bboxes))
        for bbox in bboxes:
            # bbox: x1, y1, x2, y2, ?, left_x, left_y, right_x, right_y, nose_x, nose_y, left_mouth_x, left_mouth_y, right_mouth_x, right_mouth_y
            # remove faces with too small eye distance: side faces or too small faces
            eye_dist = np.linalg.norm([left_eye[0] - right_eye[0], left_eye[1] - right_eye[1]])
            if eye_dist_threshold is not None and (eye_dist < eye_dist_threshold):
                print(f"{eye_dist} < {eye_dist_threshold}")
                continue

            if self.template_3points:
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
            else:
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
            self.all_landmarks_5.append(landmark)
            self.det_faces.append(bbox[0:5])
        if len(self.det_faces) == 0:
            return 0
        if only_keep_largest:
            h, w, _ = self.input_img.shape
            self.det_faces, largest_idx = get_largest_face(self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[largest_idx]]
        elif only_center_face:
            h, w, _ = self.input_img.shape
            self.det_faces, center_idx = get_center_face(self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[center_idx]]

        # pad blurry images
        if self.pad_blur:
            self.pad_input_imgs = []
            for landmarks in self.all_landmarks_5:
                # get landmarks
                eye_left = landmarks[0, :]
                eye_right = landmarks[1, :]
                eye_avg = (eye_left + eye_right) * 0.5
                mouth_avg = (landmarks[3, :] + landmarks[4, :]) * 0.5
                eye_to_eye = eye_right - eye_left
                eye_to_mouth = mouth_avg - eye_avg

                # Get the oriented crop rectangle
                # x: half width of the oriented crop rectangle
                x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
                #  - np.flipud(eye_to_mouth) * [-1, 1]: rotate 90 clockwise
                # norm with the hypotenuse: get the direction
                x /= np.hypot(*x)  # get the hypotenuse of a right triangle
                rect_scale = 1.5
                x *= max(np.hypot(*eye_to_eye) * 2.0 * rect_scale, np.hypot(*eye_to_mouth) * 1.8 * rect_scale)
                # y: half height of the oriented crop rectangle
                y = np.flipud(x) * [-1, 1]

                # c: center
                c = eye_avg + eye_to_mouth * 0.1
                # quad: (left_top, left_bottom, right_bottom, right_top)
                quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
                # qsize: side length of the square
                qsize = np.hypot(*x) * 2
                border = max(int(np.rint(qsize * 0.1)), 3)

                # get pad
                # pad: (width_left, height_top, width_right, height_bottom)
                pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                       int(np.ceil(max(quad[:, 1]))))
                pad = [
                    max(-pad[0] + border, 1),
                    max(-pad[1] + border, 1),
                    max(pad[2] - self.input_img.shape[0] + border, 1),
                    max(pad[3] - self.input_img.shape[1] + border, 1)
                ]

                if max(pad) > 1:
                    # pad image
                    pad_img = np.pad(self.input_img, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
                    # modify landmark coords
                    landmarks[:, 0] += pad[0]
                    landmarks[:, 1] += pad[1]
                    # blur pad images
                    h, w, _ = pad_img.shape
                    y, x, _ = np.ogrid[:h, :w, :1]
                    mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                                       np.float32(w - 1 - x) / pad[2]),
                                      1.0 - np.minimum(np.float32(y) / pad[1],
                                                       np.float32(h - 1 - y) / pad[3]))
                    blur = int(qsize * blur_ratio)
                    if blur % 2 == 0:
                        blur += 1
                    blur_img = cv2.boxFilter(pad_img, 0, ksize=(blur, blur))
                    # blur_img = cv2.GaussianBlur(pad_img, (blur, blur), 0)

                    pad_img = pad_img.astype('float32')
                    pad_img += (blur_img - pad_img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                    pad_img += (np.median(pad_img, axis=(0, 1)) - pad_img) * np.clip(mask, 0.0, 1.0)
                    pad_img = np.clip(pad_img, 0, 255)  # float32, [0, 255]
                    self.pad_input_imgs.append(pad_img)
                else:
                    self.pad_input_imgs.append(np.copy(self.input_img))

        return len(self.all_landmarks_5)
    def clean_all(self):
        self.cropped_small_faces = []
        return super().clean_all()

def create_face_helper(face_size=512):
    face_helper = LandmarksFaceRestoreHelper(
            1,
            face_size=face_size,
            crop_ratio=(1, 1),
            save_ext='png',
            current_smooth_weight=0.2,
    )
    return face_helper

if __name__ == '__main__':
    import pickle
    face_helper = create_face_helper()
    video=cv2.VideoCapture('test_paste/input1.mp4')
    f=open('test_paste/source.pkl','rb')
    video_landmarks = pickle.load(f)

    for i in range(len(video_landmarks)):
        _,frame=video.read()
        frame_landmark=video_landmarks[i]
        face_helper.clean_all()
        face_helper.read_image(frame)
        face_helper.get_face_landmarks_3(frame_landmark, only_keep_largest=True, eye_dist_threshold=5)
        face_helper.align_warp_face()

        assert len(face_helper.cropped_faces) != 0
        

        affine_matrix = face_helper.affine_matrices[0]
        large_face = face_helper.cropped_faces[0]
        face = face_helper.cropped_small_faces[0]
        affine_landmarks = (np.matmul(affine_matrix[:, :2], frame_landmark.T) + affine_matrix[:, 2:]).T
        # cv2.fillPoly(large_face,[affine_landmarks.astype(np.int32)],(255,255),1)
        cv2.imwrite(f'frame/{i}.png',large_face)