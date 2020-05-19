import os
import random
import codecs

from sklearn.model_selection import train_test_split


def get_file_names_recursion(path, file_names):
    """ 递归读取输入路径下的所有文件，file_names会递归更新.

        @params:
            path - 待递归检索的文件夹路径.
            file_names - 待输出结果的文件名列表.

        @return:
            On success - 无返回值，文件输出至file_names中.
    """
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            get_file_names_recursion(file_path, file_names)
        else:
            if '.db' not in file_path:
                file_names.append(file_path)


def output_data_records(data_records, file_name):
    """输出数据记录"""
    with codecs.open(file_name, mode='w', encoding='utf8') as fw:
        for line in data_records:
            fw.write('{}\n'.format(line))


if __name__ == "__main__":
    """训练数据预处理"""
    real_face_path = './data/NUAA/ClientFace'  # 真实
    spoof_face_path = './data/NUAA/ImposterFace'  # 伪造
    more_real_face_path = './data/NUAA/more_ClientFace'  # 补充照片真实
    more_spoof_face_path = './data/NUAA/more_ImposterFace'  # 补充照片伪造
    # 读取数据文件
    real_face_files = list()
    spoof_face_files = list()
    get_file_names_recursion(real_face_path, real_face_files)
    print('real_face_files:{}'.format(len(real_face_files)))
    get_file_names_recursion(spoof_face_path, spoof_face_files)
    print('spoof_face_files:{}'.format(len(spoof_face_files)))
    # 采样
    real_face_files = random.sample(real_face_files, 500)
    spoof_face_files = random.sample(spoof_face_files, 500)
    get_file_names_recursion(more_real_face_path, real_face_files)
    print('real_face_files:{}'.format(len(real_face_files)))
    get_file_names_recursion(more_spoof_face_path, spoof_face_files)
    print('spoof_face_files:{}'.format(len(spoof_face_files)))
    # 构造训练数据
    real_face_files = [file_name.replace('\\', '/').replace('./data/NUAA/', '') for file_name in real_face_files]
    spoof_face_files = [file_name.replace('\\', '/').replace('./data/NUAA/', '') for file_name in spoof_face_files]
    data_list = list()
    data_list.extend(['{},1'.format(file_name) for file_name in real_face_files])
    data_list.extend(['{},0'.format(file_name) for file_name in spoof_face_files])
    train_data, val_data = train_test_split(data_list, test_size=0.2, shuffle=True)
    val_data, test_data = train_test_split(val_data, test_size=0.5, shuffle=True)
    print('数据切分结果： all:{}, train:{}, val:{}, test:{}'.format(len(data_list), len(train_data), len(val_data), len(test_data)))
    train_file_name = './data/NUAA/train.txt'
    output_data_records(train_data, train_file_name)
    val_file_name = './data/NUAA/val.txt'
    output_data_records(val_data, val_file_name)
    test_file_name = './data/NUAA/test.txt'
    output_data_records(test_data, test_file_name)
