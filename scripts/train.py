import os
import sys
import time

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import faceAntiSpoofing  # noqa
print('faceAntiSpoofing module path :{}'.format(faceAntiSpoofing.__file__))  # 输出测试模块文件位置

from faceAntiSpoofing import datasets  # noqa
from faceAntiSpoofing import models  # noqa


if __name__ == "__main__":
    """模型训练"""
    PROJECT_DIR = './data'
    INPUT_SIZE = 128
    BATCH_SIZE = 32
    EPOCHS = 5
    CLASS_NAMES = ['fake', 'live']
    CLASS_NUM = len(CLASS_NAMES)

    NUAA_DATA_DIR = os.path.join(PROJECT_DIR, 'NUAA')
    MODEL_DIR = os.path.join(PROJECT_DIR, 'model')
    LOG_DIR = os.path.join(PROJECT_DIR, 'log')
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    trained_model_file = os.path.join(MODEL_DIR, 'fas_model.h5')
    fine_tune_model_file = os.path.join(MODEL_DIR, 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model_file_name = os.path.join(MODEL_DIR, 'fas_model_{}.h5'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
    print('model_file_name:{}'.format(model_file_name))
    dataset = datasets.NUAA(NUAA_DATA_DIR, batch_size=BATCH_SIZE, input_size=INPUT_SIZE, class_num=CLASS_NUM)
    net = models.FasNet(dataset, CLASS_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE,
                        fine_tune_model_file=fine_tune_model_file)
    net.train(model_file_name, MODEL_DIR, LOG_DIR, max_epoches=EPOCHS, trained_model_file=trained_model_file)
    net.predict()
