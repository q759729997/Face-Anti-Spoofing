import os

import numpy as np
import cv2
import dlib
from imutils import face_utils

from models import FasNet


def test_photo(model, img_file, out_img_file, predictor, detector):
    frames = []

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

    frame = cv2.imread(img_file)
    face_rects = detector(frame, 0)

    if len(face_rects) > 0:
        shape = predictor(frame, face_rects[0])
        shape = face_utils.shape_to_np(shape)
    else:
        raise Exception('shape None')
    # print('shape:{}'.format(shape))

    input_img_hsv, input_img_ycrcb, start_y, end_y, start_x, end_x = crop_face_loosely(
        shape, frame, INPUT_SIZE)

    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), thickness=2)

    frames.append({'hsv': input_img_hsv, 'yuv': input_img_ycrcb})
    if len(frames) == 1:
        # print(shape[30])
        pred = model.test_online(frames)
        print(pred, img_file)

        # print(pred)
        idx = np.argmax(pred[0])
        if pred[0][0] < 0.85:
            idx = 1

        text = CLASS_NAMES[idx] + ":" + str(pred[0][idx])
        print(text)
        cv2.putText(frame, text, (start_x, start_y),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
        frames = []

        # cv2.imshow("frame", frame)


if __name__ == "__main__":
    CLASS_NAMES = ['fake', 'live']
    CLASS_NUM = len(CLASS_NAMES)
    model_root_path = './data/model'
    fine_tune_model_file = os.path.join(
        model_root_path, 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    dataset, BATCH_SIZE, INPUT_SIZE = None, 32, 128
    model = FasNet(dataset,
                   CLASS_NUM,
                   batch_size=BATCH_SIZE,
                   input_size=INPUT_SIZE,
                   fine_tune_model_file=fine_tune_model_file)
    model_file = os.path.join(model_root_path, 'fas_model.h5')
    model.train(model_file,
                model_root_path,
                './data/',
                max_epoches=10,
                load_weight=True)
    image_names = [
        'qiaoyongtian_true_1',
        'qiaoyongtian_true_2',
        'lijiale_true_1',
        'lijiale_true_2',
        'zhaoshengao_true_1',
        'zhaoshengao_true_2',
        'xiaoqi_true',
        'xiaoqi_false',
        'qiaoyongtian_false_1',
        'qiaoyongtian_false_2',
        'lijiale_false_1',
        'lijiale_false_2',
        'zhaoshengao_false_1',
        'zhaoshengao_false_2',
    ]
    face_landmark_path = './data/model/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)
    for image_name in image_names:
        img_file = os.path.join('./data/images', '{}.jpg'.format(image_name))
        out_img_file = os.path.join('./data/images',
                                    'masked_{}.jpg'.format(image_name))
        test_photo(model, img_file, out_img_file, predictor, detector)
    """
    [[4.6944007e-04 9.9953055e-01]] ./data/images/qiaoyongtian_true_1.jpg
    live:0.99953055
    [[0.03954957 0.9604505 ]] ./data/images/qiaoyongtian_true_2.jpg
    live:0.9604505
    [[0.05171567 0.9482844 ]] ./data/images/lijiale_true_1.jpg
    live:0.9482844
    [[0.22602794 0.7739721 ]] ./data/images/lijiale_true_2.jpg
    live:0.7739721
    [[0.22733411 0.7726659 ]] ./data/images/zhaoshengao_true_1.jpg
    live:0.7726659
    [[0.6416532  0.35834682]] ./data/images/zhaoshengao_true_2.jpg
    live:0.35834682

    [[0.9985216  0.00147841]] ./data/images/qiaoyongtian_false_1.jpg
    fake:0.9985216
    [[0.0371988 0.9628012]] ./data/images/qiaoyongtian_false_2.jpg
    live:0.9628012
    [[0.9989931  0.00100685]] ./data/images/lijiale_false_1.jpg
    fake:0.9989931
    [[0.99896145 0.00103852]] ./data/images/lijiale_false_2.jpg
    fake:0.99896145
    [[0.99844426 0.00155578]] ./data/images/zhaoshengao_false_1.jpg
    fake:0.99844426
    [[0.99140745 0.00859255]] ./data/images/zhaoshengao_false_2.jpg
    fake:0.99140745
    """
