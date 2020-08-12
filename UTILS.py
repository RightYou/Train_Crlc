import math
import os
import random
import re
import threading

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.linalg import hadamard

from TRAIN_CRLC import BATCH_SIZE
from TRAIN_CRLC import PATCH_SIZE


class Reader(threading.Thread):
    def __init__(self, file_name, id, input_list, gt_list):
        super(Reader, self).__init__()
        self.file_name = file_name
        self.id = id
        self.input_list = input_list
        self.gt_list = gt_list

    def run(self):
        input_image = c_getYdata(self.file_name[0])
        gt_image = c_getYdata(self.file_name[1])
        in_ = []
        gt_ = []
        for j in range(BATCH_SIZE // 8):
            input_imgY, gt_imgY = crop(input_image, gt_image, PATCH_SIZE[0], PATCH_SIZE[1], "ndarray")
            input_imgY = normalize(input_imgY)  # 这里的input_imgY是64x64的
            gt_imgY = normalize(gt_imgY)

            in_.append(input_imgY)
            gt_.append(gt_imgY)

        self.input_list[self.id] = in_
        self.gt_list[self.id] = gt_


def normalize(x):
    x = x / 255.
    return truncate(x, 0., 1.)


def denormalize(x):
    x = x * 255.
    return truncate(x, 0., 255.)


def truncate(input, min, max):
    input = np.where(input > min, input, min)
    input = np.where(input < max, input, max)
    return input


def load_file_list(directory):
    list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_name = os.path.join(root, file)
            if file_name.split(".")[-1] == "yuv":
                list.append(file_name)
    return sorted(list)


def get_train_list(lowList, highList):
    assert len(lowList) % len(highList) == 0, "low:%d, high:%d" % (len(lowList), len(highList))
    train_list = []
    for i in range(len(lowList)):
        qp = lowList[i].split("\\")[-2].split("qp")[1]

        if 56 <= int(qp) <= 65:
            train_list.append([lowList[i], highList[i % len(highList)]])
    return train_list


def prepare_nn_data(train_list):
    thread_num = 8
    batchSizeRandomList = random.sample(range(0, len(train_list)), thread_num)
    input_list = [0 for i in range(thread_num)]
    gt_list = [0 for i in range(thread_num)]
    t = []
    for i in range(thread_num):
        t.append(Reader(train_list[batchSizeRandomList[i]], i, input_list, gt_list))
    for i in range(thread_num):
        t[i].start()
    for i in range(thread_num):
        t[i].join()
    input_list = np.reshape(input_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    gt_list = np.reshape(gt_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))

    return input_list, gt_list


def getWH(yuvfileName):  # Train
    deyuv = re.compile(r'(.+?)\.')  #
    deyuvFilename = deyuv.findall(yuvfileName)[0]  # 去yuv后缀的文件名
    if os.path.basename(deyuvFilename).split("_")[0].isdigit():
        wxh = os.path.basename(deyuvFilename).split('_')[1]
    else:
        wxh = os.path.basename(deyuvFilename).split('_')[1]
    w, h = wxh.split('x')
    return int(w), int(h)


def getYdata(path, size):
    w = size[0]
    h = size[1]
    Yt = np.zeros([h, w], dtype="uint8", order='C')
    with open(path, 'rb') as fp:
        fp.seek(0, 0)
        Yt = fp.read()
        tem = Image.frombytes('L', [w, h], Yt)
        Yt = np.asarray(tem, dtype='float32')
    return Yt


def c_getYdata(path):
    return getYdata(path, getWH(path))


def crop(input_image, gt_image, patch_width, patch_height, img_type):
    assert type(input_image) == type(gt_image), "types are different."
    # return a ndarray object
    if img_type == "ndarray":
        in_row_ind = random.randint(0, input_image.shape[0] - patch_width)
        in_col_ind = random.randint(0, input_image.shape[1] - patch_height)

        input_cropped = input_image[in_row_ind:in_row_ind + patch_width, in_col_ind:in_col_ind + patch_height]
        gt_cropped = gt_image[in_row_ind:in_row_ind + patch_width, in_col_ind:in_col_ind + patch_height]

    # return an "Image" object
    elif img_type == "Image":
        in_row_ind = random.randint(0, input_image.size[0] - patch_width)
        in_col_ind = random.randint(0, input_image.size[1] - patch_height)

        input_cropped = input_image.crop(
            box=(in_row_ind, in_col_ind, in_row_ind + patch_width, in_col_ind + patch_height))
        gt_cropped = gt_image.crop(box=(in_row_ind, in_col_ind, in_row_ind + patch_width, in_col_ind + patch_height))

    return input_cropped, gt_cropped


def psnr(hr_image, sr_image, max_value=255.0):
    eps = 1e-10
    if ((type(hr_image) == type(np.array([]))) or (type(hr_image) == type([]))):
        hr_image_data = np.asarray(hr_image, 'float32')
        sr_image_data = np.asarray(sr_image, 'float32')

        diff = sr_image_data - hr_image_data
        mse = np.mean(diff * diff)
        mse = np.maximum(eps, mse)
        return float(10 * math.log10(max_value * max_value / mse))
    else:
        assert len(hr_image.shape) == 4 and len(sr_image.shape) == 4
        diff = hr_image - sr_image
        mse = tf.reduce_mean(tf.square(diff))
        mse = tf.maximum(mse, eps)
        return 10 * tf.log(max_value * max_value / mse) / math.log(10)
