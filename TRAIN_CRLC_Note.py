import argparse
import time
from random import shuffle  # 随机排序，用于打乱数据集
from UTILS import *  # 工具包，有一些处理数据的方法，同时import了一些库
from CRLC_v24 import crlc_model as model  # CNN模型

tf.logging.set_verbosity(tf.logging.WARN)  # 默认情况下，tensorflow在WARN的日志记录级别进行配置，但是在跟踪模型训练时，需要将级别调整为INFO
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 选择GPU的使用，一般都是0

# ----- 一些基本参数的设置 ----- #
EXP_DATA = 'CRLCv24_I_QP47-56_20072301'  # 模型命名

LOW_DATA_PATH = r"G:\Kong\av1_deblock_nocdefLr"  # 训练集的位置
HIGH_DATA_PATH = r"G:\Kong\div2k_train_hr_yuv"  # 训练集标签的位置

LOG_PATH = "./logs/%s/" % EXP_DATA  # log文件的存放位置
CKPT_PATH = "./checkpoints/%s/" % EXP_DATA  # 训练出的模型的存放位置
SAMPLE_PATH = "./samples/%s/" % EXP_DATA  # 存放结果的图片，目前用不上

PATCH_SIZE = (64, 64)  # 输入到CNN中的图片的尺寸。CNN内核一次只处理一个patch，而不是整个图片
BATCH_SIZE = 64  # 每批样本的大小，整个训练集不是一次性全部喂入CNN的，需要分批次

BASE_LR = 1e-3  # 基本学习率，一开始的学习率
LR_DECAY_RATE = 0.5  # 学习率衰减指数，决定了学习率一次衰减的比率
LR_DECAY_STEP = 25  # 学习率衰减步长，衰减速度，也就是每迭代多少轮，学习率衰减一次
MAX_EPOCH = 500  # 多少代训练，整个训练集重复训练的次数，最外面的大循环，每次都会生成一个模型

# 通过命令行的方式输入模型的路径，用于断点续训，使用方法为在终端输入
# python.exe TRAIN.py --model_path C:/Users/User/模型的路径
# 或者填入PyCharm的Configuration的parameter中。
parser = argparse.ArgumentParser()  # 创建了一个命令行解析器
parser.add_argument("--model_path")  # 添加参数
args = parser.parse_args()  # 解析参数
model_path = args.model_path  # model_path存放的是最新生成的model的路径，用于断点续训
# model_path = r"D:\wzy\Train\out\checkpoints\CNN_QP53_2020\CNN_QP53_2020_261_51.20.ckpt"

# 超参数剪枝操作，模型剪枝
# https://blog.csdn.net/lai_cheng/article/details/90643100
# https://blog.csdn.net/xue_csdn/article/details/105220985
# Get, Print, and Edit Pruning Hyper parameters
# pruning_hparams = pruning.get_pruning_hparams()
# print("Pruning Hyper parameters:", pruning_hparams)
# # Change hyperparameters to meet our needs
# pruning_hparams.begin_pruning_step = 0
# pruning_hparams.end_pruning_step = 250
# pruning_hparams.pruning_frequency = 1
# pruning_hparams.sparsity_function_end_step = 250
# pruning_hparams.target_sparsity = 0.9
# # Create a pruning object using the pruning specification, sparsity seems to have priority over the hparam
# p = pruning.Pruning(pruning_hparams, global_step=2, sparsity=0.9)
# prune_op = p.conditional_mask_update_op()

if __name__ == '__main__':
    # 初始准备
    start = time.time()  # 获取当前时间，用于记录处理时间
    # 获取训练集，讲数据和标签配对
    train_list = get_train_list(load_file_list(LOW_DATA_PATH), load_file_list(HIGH_DATA_PATH))
    print(len(train_list))

    # 设置一些张量
    with tf.name_scope('input_scope'):
        train_input = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
        train_gt = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))

        # shared_model = tf.make_template('shared_model', model)  # 给定一个任意函数,将其包装,以便它进行变量共享
        train_output = model(train_input)  # 传入cnn中训练，返回输出
        R = tf.reshape(train_output, [64, 64 * 64, tf.shape(train_output)[-1]])

    with tf.name_scope('loss_scope'), tf.device("/gpu:0"):
        # 使用最小二乘法获得A
        A = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(R, perm=[0, 2, 1]), R)),
                                tf.transpose(R, perm=[0, 2, 1])),
                      tf.reshape(tf.subtract(train_gt, train_input), [64, 64 * 64, 1]))
        # A的转置 = (R的转置*R)的逆 * R的转置 * r   r = s - x 就是标签减去预测值
        # tf.matrix_inverse：矩阵的逆
        # tf.transpose: 将a进行转置，并且根据perm参数重新排列输出维度

        # 设置损失函数
        loss = tf.reduce_sum(-(tf.matmul(
            tf.matmul(tf.transpose(tf.reshape(tf.subtract(train_gt, train_input), [64, 64 * 64, 1]), perm=[0, 2, 1]),
                      R), A)))  # 损失函数用MSE均值平方差
        # loss = 求和-(r的转置 * R * A)

        avg_loss = tf.placeholder('float32')  # 平均损失
        tf.summary.scalar("avg_loss", avg_loss)  # 用来显示标量信息，一般在画loss时会用到这个函数。

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(BASE_LR, global_step, LR_DECAY_STEP * 1000, LR_DECAY_RATE,
                                               staircase=True)
    tf.summary.scalar("learning rate", learning_rate)

    # 设置优化器
    optimizer_adam = tf.train.AdamOptimizer(learning_rate, 0.9)
    opt_adam = optimizer_adam.minimize(loss, global_step=global_step)

    # 另外一种优化器，第三行原代码是打开的，但感觉没什么用就注释了
    # optimizer_SGD = tf.train.GradientDescentOptimizer(learning_rate)
    # opt_SGD = optimizer_SGD.minimize(loss, global_step=global_step)
    # opt_SGD = tf.train.GradientDescentOptimizer(0.5).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=0)  # 返回一个类，用于保存和加载模型

    # 用于一边训练一边测试吧
    # with tf.name_scope('testInput_scope'):
    #     test_input = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    #     test_output, _ = shared_model(test_input)
    #     test_input_data, test_gt_data, test_cbcr_data = prepare_nn_data(train_list, 56)

    # 设置tf.Session的运算方式，同时设置了GPU的使用
    config = tf.ConfigProto(allow_soft_placement=True)  # 当运行设备不满足要求时，会自动分配GPU或者CPU。
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # GPU内存使用最大比例
    config.gpu_options.allow_growth = True  # 运行GPU内存自增长

    # 开始运行Session
    with tf.Session(config=config) as sess:
        # 准备需要的文件夹
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        if not os.path.exists(os.path.dirname(CKPT_PATH)):
            os.makedirs(os.path.dirname(CKPT_PATH))
        if not os.path.exists(SAMPLE_PATH):
            os.makedirs(SAMPLE_PATH)

        merged = tf.summary.merge_all()  # 自动管理
        file_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
        sess.run(tf.global_variables_initializer())  # 对所有变量初始化
        last_epoch = 0

        # 如果给了断点的模型路径，就加载模型，继续训练
        if model_path:
            print("restore model...", model_path)
            saver.restore(sess, model_path)  # 恢复模型
            last_epoch = int(os.path.basename(model_path).split(".")[0].split("_")[-2])
            print(last_epoch)
        print("prepare_time:", time.time() - start)

        for epoch in range(last_epoch, last_epoch + MAX_EPOCH):
            shuffle(train_list)  # 对训练数据乱序，提高训练效果

            total_g_loss, n_iter = 0, 0

            epoch_time = time.time()  # 这一代训练的开始时间
            total_get_data_time, total_network_time = 0, 0
            for idx in range(1000):
                get_data_time = time.time()  # 开始加载数据的时间
                input_data, gt_data = prepare_nn_data(train_list)  # 从列表取图像，提取图片的函数
                # temp = total_get_data_time
                total_get_data_time += (time.time() - get_data_time)  # 总的加载数据时间=sum(现在的时间-开始加载数据的时间)
                # print(idx, total_get_data_time - temp)
                network_time = time.time()  # 网络开始的时间
                feed_dict = {train_input: input_data, train_gt: gt_data}  # 以字典的形式喂入数据
                _, l, output, g_step = sess.run([opt_adam, loss, train_output, global_step], feed_dict=feed_dict)
                # print(A[0,0,0,])  # 不知道有啥用
                total_network_time += (time.time() - network_time)  # 网络总的运行时间=sum(现在的时间-网络开始的时间)
                total_g_loss += l
                n_iter += 1
                # print(output[0])
                # file_writer.add_summary(summary, g_step)
                del input_data, gt_data, output

            lr, summary = sess.run([learning_rate, merged], {avg_loss: total_g_loss / n_iter})
            file_writer.add_summary(summary, epoch)
            # print("Epoch: [%4d/%4d]  time: %4.4f\tloss: %.8f\tlr: %.8f" %
            #       (epoch, MAX_EPOCH, time.time() - epoch_time, total_g_loss / n_iter, lr))
            tf.logging.warning(
                "Epoch: [%4d/%4d]  time: %4.4f\t loss: %.8f\t lr: %.16f\t"
                "total_get_data_time: %8f\t total_network_time: %8f"
                % (epoch, last_epoch + MAX_EPOCH, time.time() - epoch_time, total_g_loss / n_iter, lr,
                   total_get_data_time, total_network_time)
            )  # 通过tf输出日志的方式，输出训练的相关情况

            # 用于一边训练一边测试吧
            # test_out = sess.run(test_output, feed_dict={test_input: test_input_data})
            # test_out = denormalize(test_out)
            # save_images(test_out, test_cbcr_data, [8, 8], os.path.join(SAMPLE_PATH,"epoch%s.png"%epoch))

            # if ((epoch + 1) % 10 == 0):  # 每训练十代，保存一次模型
            saver.save(sess, os.path.join(CKPT_PATH, "%s_%03d_%.2f.ckpt" % (EXP_DATA, epoch, total_g_loss / n_iter)))
