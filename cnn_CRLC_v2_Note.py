import sys
import numpy.linalg as lg
import matplotlib.pyplot as plt  # 用于绘制A的分布图，便于量化
from UTILS import *
from RVDSR_WARN_2 import model as model

if not hasattr(sys, 'argv'):  # 如果没有命令，就把命令置为空，用处未知
    sys.argv = ['']
np.set_printoptions(threshold=np.inf)  # 用于数组完全显示

format_pt = "{0:^30}\t{1:^10}\t{2:^10}"  # 用于格式化输出

# init函数使用，模型权重文件路径，通过字典的方式便于获取
model_set = {
    "QP7~16_C2": r"E:\best\CRLCv24_real_I_QP7~16_C2\CRLCv24_real_I_QP7~16_C2_20040101_097_-0.49.ckpt",
    "QP17~26_C2": r"E:\best\CRLCv24_real_I_QP17~26_C2\CRLCv24_real_I_QP17~26_C2_20040201_107_-1.42.ckpt",
    "QP27~36_C2": r"E:\best\CRLCv24_real_I_QP27~36_C2\CRLCv24_real_I_QP27~36_C2_20040701_165_-14.12.ckpt",
    "QP37~46_C2": r"E:\best\CRLCv24_real_I_QP37~46_C2\CRLCv24_real_I_QP37~46_C2_20040301_178_-10.60.ckpt",
    "QP47~56_C2": r"E:\best\CRLCv24_real_I_QP47~56_C2\CRLCv24_real_I_QP47~56_C2_20040301_194_-21.95.ckpt",
    "QP57~66_C2": r"E:\best\CRLCv24_real_I_QP57~66_C2\CRLCv24_real_I_QP57~66_C2_20040301_190_-39.37.ckpt",
    "test": r""
}

# predict函数使用，量化范围集合
range_set = {
    "range8": {
        "QP7~16_C2": (512, 1, -5),
        "QP17~26_C2": (256, 0, -1),
        "QP27~36_C2": (512, -7, -7),
        "QP37~46_C2": (256, -5, -7),
        "QP47~56_C2": (256, -8, -4),
        "QP57~64_C2": (128, -6, -7)
    },
    "range16": {
        # crlc的量化
        "QP7~16_C2": (1024, 0, -7),
        "QP17~26_C2": (512, 2, -3),
        # "QP27~36_C2": (1024, -13, -15),
        # "QP37~46_C2": (512, -13, -13),
        # "QP47~56_C2": (512, -15, -10),
        # "QP57~64_C2": (256, -9, -10)

        # vdsr_warn的量化
        "QP27~36_C2": (1024, 0, -15),
        "QP37~46_C2": (512, 0, -11),
        "QP47~56_C2": (128, -6, -11),
        "QP57~64_C2": (512, -7, -13)
    },
    "range32": {
        "QP7~16_C2": (2048, 5, -24),
        "QP17~26_C2": (1024, 2, -3),
        "QP27~36_C2": (2048, -23, -28),
        "QP37~46_C2": (1024, -23, -23),
        "QP47~56_C2": (1024, -30, -10),
        "QP57~64_C2": (512, -16, -20)
    },
    "range64": {
        "QP7~16_C2": (2048, 15, -30),
        "QP17~26_C2": (2048, 10, -10),
        "QP27~36_C2": (4096, -45, -60),
        "QP37~46_C2": (2048, -40, -45),
        "QP47~56_C2": (2048, -60, -30),
        "QP57~64_C2": (1024, -40, -40)
    }
}

# 设置tf.Session的运算方式，同时设置了GPU的使用
config = tf.ConfigProto(allow_soft_placement=True)  # 当运行设备不满足要求时，会自动分配GPU或者CPU。
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # GPU内存使用最大比例，这里设置为1，就会限制共享内存的使用，所以可以注释
config.gpu_options.allow_growth = True  # 运行GPU内存自增长

# 定义用来调用模型的全局变量
global CNN_Model

# 用于统计A0和A1的值的分布
a0_num = dict()
a1_num = dict()


# 准备测试数据
def prepare_test_data(fileOrDir):
    original_ycbcr = []
    imgCbCr = []
    gt_y = []
    fileName_list = []
    # The input is a single file.
    if type(fileOrDir) is str:
        fileName_list.append(fileOrDir)

        # w, h = get_w_h(fileOrDir)
        # imgY = getYdata(fileOrDir, [w, h])
        imgY = c_get_y_data(fileOrDir)
        imgY = normalize(imgY)

        imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
        original_ycbcr.append([imgY, imgCbCr])

    ##The input is one directory of test images.
    elif len(fileOrDir) == 1:
        fileName_list = load_file_list(fileOrDir)
        for path in fileName_list:
            # w, h = get_w_h(path)
            # imgY = getYdata(path, [w, h])
            imgY = c_get_y_data(path)
            imgY = normalize(imgY)

            imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
            original_ycbcr.append([imgY, imgCbCr])

    ##The input is two directories, including ground truth.
    elif len(fileOrDir) == 2:
        fileName_list = load_file_list(fileOrDir[0])
        test_list = get_test_list(load_file_list(fileOrDir[0]), load_file_list(fileOrDir[1]))
        for pair in test_list:
            filesize = os.path.getsize(pair[0])
            picsize = get_w_h(pair[0])[0] * get_w_h(pair[0])[0] * 3 // 2
            numFrames = filesize // picsize
            # if numFrames ==1:
            or_imgY = c_get_y_data(pair[0])
            gt_imgY = c_get_y_data(pair[1])

            # normalize
            or_imgY = normalize(or_imgY)

            or_imgY = np.resize(or_imgY, (1, or_imgY.shape[0], or_imgY.shape[1], 1))
            gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1], 1))

            ## act as a placeholder
            or_imgCbCr = 0
            original_ycbcr.append([or_imgY, or_imgCbCr])
            gt_y.append(gt_imgY)
            # else:
            #     while numFrames>0:
            #         or_imgY =getOneFrameY(pair[0])
            #         gt_imgY =getOneFrameY(pair[1])
            #         # normalize
            #         or_imgY = normalize(or_imgY)
            #
            #         or_imgY = np.resize(or_imgY, (1, or_imgY.shape[0], or_imgY.shape[1], 1))
            #         gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1], 1))
            #
            #         ## act as a placeholder
            #         or_imgCbCr = 0
            #         original_ycbcr.append([or_imgY, or_imgCbCr])
            #         gt_y.append(gt_imgY)
    else:
        print("Invalid Inputs.")
        exit(0)

    return original_ycbcr, gt_y, fileName_list


class Predict:
    input_tensor = None
    output_tensor = None
    model = None
    r = None
    gt = None
    R = None
    R_out = None

    def __init__(self, model, modelpath):
        self.graph = tf.Graph()  # 为每个类(实例)单独创建一个graph
        self.model = model
        with self.graph.as_default():
            self.input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
            self.gt = tf.placeholder(tf.float32, shape=(1, None, None, 1))
            # self.r = tf.reshape(tf.subtract(self.gt, self.input_tensor),
            #                     [1, tf.shape(self.input_tensor)[1] * tf.shape(self.input_tensor)[2], 1])
            # self.R = tf.make_template('shared_model', self.model)(self.input_tensor)
            self.R = self.model(self.input_tensor)
            # R_ = tf.reshape(self.R,
            #                 [1, tf.shape(self.input_tensor)[1] * tf.shape(self.input_tensor)[2],tf.shape(self.R)[-1]])
            # r_ = tf.reshape(tf.subtract(self.gt, self.input_tensor),
            #                 [1, tf.shape(self.input_tensor)[1] * tf.shape(self.input_tensor)[2], 1])
            #
            # self.A = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(R_, perm=[0, 2, 1]), R_)),
            #                         tf.transpose(R_, perm=[0, 2, 1])), r_)
            # self.A=tf.round(tf.multiply(tf.reshape(self.A,(1,tf.shape(self.R)[-1])),128))
            # A0 = self.A
            # A1 = self.A
            # A0 = tf.clip_by_value(A0, -8, 23)
            # A1 = tf.clip_by_value(A1, -16, 15)
            # self.A=[A0[0,0],A1[0,1]]
            # self.output_tensor = tf.add(
            #     (tf.reduce_sum(tf.multiply(self.R, tf.multiply(self.A, 1 / 128)), axis=3, keep_dims=True)),
            #     self.input_tensor)
            # self.output_tensor=self.R
            self.output_tensor = tf.multiply(self.R, 255)
            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=self.graph, config=config)  # 创建新的sess
        with self.sess.as_default():
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())
                self.saver.restore(self.sess, modelpath)  # 从恢复点恢复参数
                print(modelpath)

    def predict(self, fileOrDirX, fileOrDirY):
        if (isinstance(fileOrDirX, str)):
            original_ycbcr0, gt_y, fileName_list = prepare_test_data(fileOrDirX)
            original_ycbcr1, gt_y, fileName_list = prepare_test_data(fileOrDirY)
            imgX = original_ycbcr0[0][0]
            imgY = original_ycbcr1[0][0]

        elif type(fileOrDirX) is np.ndarray:
            imgX = fileOrDirX
            imgY = fileOrDirY

        elif (isinstance(fileOrDirX, list) and isinstance(fileOrDirY, list)):
            # print("model.predict", type(fileOrDirX), type(fileOrDirY))
            # print(len(fileOrDirX),len(fileOrDirX[0]))
            fileOrDirX = np.asarray(fileOrDirX, dtype='float32')
            fileOrDirY = np.asarray(fileOrDirY, dtype='float32')
            imgX = normalize(np.reshape(fileOrDirX, (1, len(fileOrDirX), len(fileOrDirX[0]), 1)))
            imgY = normalize(np.reshape(fileOrDirY, (1, len(fileOrDirY), len(fileOrDirY[0]), 1)))
            # imgX = np.reshape(fileOrDirX, (1, len(fileOrDirX), len(fileOrDirX[0]), 1))
            # imgY = np.reshape(fileOrDirY, (1, len(fileOrDirY), len(fileOrDirY[0]), 1))

        else:
            imgX = None
            imgY = None

        with self.sess.as_default():
            with self.sess.graph.as_default():
                out = self.sess.run([self.output_tensor],
                                    feed_dict={self.input_tensor: imgX, self.gt: imgY})
                out = np.reshape(out, np.shape(out)[2:])
                return out


def init(frame_type, qp):
    print("----------Start to initialize the model----------")
    global CNN_Model
    qp = qp / 4  # aom传过来的是qp是0-255的，而参数配置的qp是0-64的，所以要除4
    # model: 函数本身也是对象，所以可以将函数作为参数传入另一函数并进行调用
    if qp < 17:
        CNN_Model = Predict(model, model_set["QP7~16_C2"])
    elif 17 <= qp < 27:
        CNN_Model = Predict(model, model_set["QP17~26_C2"])
    elif 27 <= qp < 37:
        CNN_Model = Predict(model, model_set["QP27~36_C2"])
    elif 37 <= qp < 47:
        CNN_Model = Predict(model, model_set["QP37~46_C2"])
    elif 47 <= qp < 57:
        CNN_Model = Predict(model, model_set["QP47~56_C2"])
    else:
        CNN_Model = Predict(model, model_set["QP57~66_C2"])
    print("-------Successfully initialized the model-------")


def predict(dgr, src, qp, block_size=256):
    qp = qp / 4
    print("当前QP为：", qp)
    A_range = 16
    # 根据qp不同，使用不同量化范围
    if qp < 17:
        scale, A0_min, A1_min = range_set["range" + str(A_range)]["QP7~16_C2"]
    elif 17 <= qp < 27:
        scale, A0_min, A1_min = range_set["range" + str(A_range)]["QP17~26_C2"]
    elif 27 <= qp < 37:
        scale, A0_min, A1_min = range_set["range" + str(A_range)]["QP27~36_C2"]
    elif 37 <= qp < 47:
        scale, A0_min, A1_min = range_set["range" + str(A_range)]["QP37~46_C2"]
    elif 47 <= qp < 57:
        scale, A0_min, A1_min = range_set["range" + str(A_range)]["QP47~56_C2"]
    else:
        scale, A0_min, A1_min = range_set["range" + str(A_range)]["QP57~64_C2"]

    global CNN_Model
    R = CNN_Model.predict(dgr, src)

    hei = np.shape(dgr)[0]
    wid = np.shape(dgr)[1]
    rows = math.ceil(float(hei) / block_size)
    cols = math.ceil(float(wid) / block_size)
    dgr = np.asarray(dgr, dtype='float32')
    src = np.asarray(src, dtype='float32')
    rec = np.zeros(np.shape(src))
    A_list = []
    for i in range(rows):
        for j in range(cols):
            if i == rows - 1:
                start_row = hei - block_size
                end_row = hei
            else:
                start_row = i * block_size
                end_row = (i + 1) * block_size
            if j == cols - 1:
                start_col = wid - block_size
                end_col = wid
            else:
                start_col = j * block_size
                end_col = (j + 1) * block_size
            if hei < block_size:
                start_row = 0
                end_row = hei
            if wid < block_size:
                start_col = 0
                end_col = wid

            # print(start_row, end_row, start_col, end_col)  # 对图像通过切片的方式进行分块处理
            sub_dgr = dgr[start_row:end_row, start_col:end_col]
            sub_src = src[start_row:end_row, start_col:end_col]
            sub_r = (sub_src - sub_dgr).flatten()
            sub_R = np.reshape(R[start_row:end_row, start_col:end_col, :],
                               ((end_col - start_col) * (end_row - start_row), np.shape(R)[-1]))
            A = np.linalg.inv(sub_R.T.dot(sub_R)).dot(sub_R.T).dot(sub_r) * scale

            # 对A量化，码率和画质的平衡
            # A = np.around(A)
            # A[0] = np.clip(A[0], A0_min, A0_min + A_range - 1)
            # A[1] = np.clip(A[1], A1_min, A1_min + A_range - 1)
            # print(A)

            # 统计A的分布
            # if A[0] not in a0_num:
            #     a0_num[A[0]] = 1
            # else:
            #     a0_num[A[0]] += 1
            #
            # if A[1] not in a1_num:
            #     a1_num[A[1]] = 1
            # else:
            #     a1_num[A[1]] += 1

            # 通过A的值组合通道还原重建图像
            sub_rec = sub_dgr + np.sum(np.multiply(R[start_row:end_row, start_col:end_col, :], A / scale), axis=2)
            rec[start_row:end_row, start_col:end_col] = sub_rec

            A_list.append(A.astype('int'))

    A_list = np.array(A_list).flatten()
    A_list = np.around(A_list)
    A_list = A_list.astype('int')
    A_list = A_list.tolist()
    # print(A_list,flush=True)

    rec = np.around(rec)
    rec = np.clip(rec, 0, 255)
    rec = rec.astype('int')
    rec = rec.tolist()
    # print("psnr：", psnr(src, rec))
    # show_img(rec)  # 显示重建图像
    return rec, A_list


def show_img(img_np):
    tem = np.asarray(img_np, dtype='uint8')
    tem = Image.fromarray(tem, 'L')
    tem.show()


def test_all_ckpt(modelPath):
    global CNN_Model
    low_img = r"F:\0wzy_Data\test_set\QP32"
    high_img = r"F:\0wzy_Data\test_set\label"
    original_ycbcr, gt_y, file_name_list = prepare_test_data([low_img, high_img])  # 加载验证集及其标签
    total_img = len(file_name_list)  # 获取验证集的总量，用于待会求psnr均值

    tem = [f for f in os.listdir(modelPath) if 'data' in f]  # 读取出其中的ckpt文件
    ckpt_files = sorted([r.split('.data')[0] for r in tem])  # 去除后缀，并排序

    max_ckpt = 0  # 最好的模型
    max_ckpt_psnr = 0  # 最大的峰值信噪比
    for ckpt in ckpt_files:
        epoch = int(ckpt.split('.')[0].split('_')[-2])  # 获取这个模型是第多少个epoch训练出来的
        # if epoch < 493:
        #     continue

        CNN_Model = Predict(model, os.path.join(modelPath, ckpt))  # CNN_Model是全局变量，已经直接传进predict了
        sum_img_psnr = 0
        img_index = [14, 17, 4, 2, 7, 10, 12, 3, 0, 13, 16, 5, 6, 1, 15, 8, 9, 11]  # 手动乱序，提高效果
        for i in img_index:
            imgY = original_ycbcr[i][0]
            gtY = gt_y[i] if gt_y else 0

            # 不知道干什么的代码
            # print(np.shape(imgY), np.shape(gtY))
            # showImg(denormalize(np.reshape(original_ycbcr[i][0], [480, 832])))
            # showImg(np.reshape(gtY, [480, 832]))
            # print(imgY.shape)
            #
            # block_size = 64
            # padding_size = 8
            # sub_imgs = zip(divide_img(imgY, block_size, padding_size),
            #                divide_img((normalize(gtY)), block_size, padding_size))
            # recs = []
            # for lowY, gY in sub_imgs:
            #     # print(type(predictor.predict(lowY, gY)))
            #     recs.append(predictor.predict(lowY, gY)[0])
            # rec = compose_img(imgY, recs, block_size, padding_size)
            # # print(psnr(np.reshape(denormalize(imgY), np.shape(rec)), np.reshape(gtY, np.shape(rec))))
            # cur_img_psnr = psnr(rec, np.reshape(gtY, np.shape(rec)))

            rec, _ = predict(denormalize(imgY)[0, :, :, 0].tolist(), gtY[0, :, :, 0].tolist(), 128, 256)
            cur_img_psnr = psnr(rec, np.reshape(gtY, np.shape(rec)))

            # 不知道干什么的代码X2
            # print(psnr(denormalize(np.reshape(imgY[:, :64, :64, :], [np.shape(imgY[:, :64, :64, :])[1],
            #                                                          np.shape(imgY[:, :64, :64, :])[2]])),
            #            np.reshape(gtY[:, :64, :64, :],
            #                       [np.shape(imgY[:, :64, :64, :])[1], np.shape(imgY[:, :64, :64, :])[2]])))
            # showImg(np.reshape(gtY[:, :64, :64, :], np.shape(rec)))
            # cur_psnr[cnnTime] = psnr(rec, np.reshape(gtY[:, :64, :64, :], np.shape(rec)))
            #
            # print(psnr(denormalize(np.reshape(imgY, [np.shape(imgY)[1], np.shape(imgY)[2]])),
            #            np.reshape(gtY, [np.shape(imgY)[1], np.shape(imgY)[2]])))

            # 对图片psnr求和
            sum_img_psnr = cur_img_psnr + sum_img_psnr
            print(format_pt.format(os.path.basename(file_name_list[i]), cur_img_psnr,
                                   psnr(denormalize(np.reshape(imgY, np.shape(rec))), np.reshape(gtY, np.shape(rec)))))

        cur_ckpt_psnr = sum_img_psnr / total_img  # 模型的psnr等于图片psnr的均值
        if cur_ckpt_psnr > max_ckpt_psnr:
            max_ckpt_psnr = cur_ckpt_psnr
            max_ckpt = epoch
        print("cur_ckpt:", epoch, " agv_psnr: ", cur_ckpt_psnr, "max ckpt:", max_ckpt, " max_psnr：", max_ckpt_psnr)


if __name__ == '__main__':
    test_all_ckpt(r"C:\Users\admin\Desktop\RVDSR_WARN_C2\QP30-36")
    # plt.bar(a0_num.keys(), a0_num.values())
    # plt.show()
    # plt.bar(a1_num.keys(), a1_num.values())
    # plt.show()
