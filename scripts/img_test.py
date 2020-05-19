import os

import cv2
import dlib
from imutils import face_utils


def crop_face_loosely(shape, img, input_size):
    x = []
    y = []
    for (_x, _y) in shape:
        x.append(_x)
        y.append(_y)

    max_x = min(max(x), img.shape[1])
    min_x = max(min(x), 0)
    max_y = min(max(y), img.shape[0])
    min_y = max(min(y), 0)

    Lx = max_x - min_x
    Ly = max_y - min_y

    Lmax = int(max(Lx, Ly))

    delta = Lmax // 2

    center_x = (max(x) + min(x)) // 2
    center_y = (max(y) + min(y)) // 2
    start_x = int(center_x - delta)
    start_y = int(center_y - delta - 30)
    end_x = int(center_x + delta)
    end_y = int(center_y + delta)

    if start_y < 0:
        start_y = 0
    if start_x < 0:
        start_x = 0
    if end_x > img.shape[1]:
        end_x = img.shape[1]
    if end_y > img.shape[0]:
        end_y = img.shape[0]

    crop_face = img[start_y:end_y, start_x:end_x]

    # cv2.imshow('crop_face', crop_face)
    img_hsv = cv2.cvtColor(crop_face, cv2.COLOR_BGR2HSV)
    img_ycrcb = cv2.cvtColor(crop_face, cv2.COLOR_BGR2YCrCb)

    img_hsv = cv2.resize(img_hsv, (input_size, input_size)) / 255.0
    img_ycrcb = cv2.resize(img_ycrcb, (input_size, input_size)) / 255.0
    return img_hsv, img_ycrcb, start_y, end_y, start_x, end_x


if __name__ == "__main__":
    """"图片测试"""
    # ./data/NUAA/ImposterFace/0001/0001_00_00_01_215.jpg
    INPUT_SIZE = 128
    img_file_name = os.path.join('./data/NUAA/ImposterFace', '0001/0001_00_00_01_215.jpg')
    frame = cv2.imread(img_file_name)
    frame = cv2.cvtColor(frame, 40)
    print(frame)
    # face_landmark_path = './data/model/shape_predictor_68_face_landmarks.dat'
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(face_landmark_path)
    # face_rects = detector(frame, 0)
    # if len(face_rects) > 0:
    #     shape = predictor(frame, face_rects[0])
    #     shape = face_utils.shape_to_np(shape)
    # else:
    #     raise Exception('shape None')
    # # print('shape:{}'.format(shape))

    # input_img_hsv, input_img_ycrcb, start_y, end_y, start_x, end_x = crop_face_loosely(
    #     shape, frame, INPUT_SIZE)
    # print(start_y, end_y, start_x, end_x)  # 21 221 34 204
