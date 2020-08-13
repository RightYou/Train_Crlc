import argparse
import time
from random import shuffle
from UTILS import *
from VDSR_WARN_C2 import model as model

tf.logging.set_verbosity(tf.logging.WARN)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set the cuda devices if you have multiple GPUs

EXP_DATA = 'VDSR_WARN_C2_I_QP47-56_200813'

LOW_DATA_PATH = r"F:\0wzy_Data\train_set\av1_deblock_nocdefLr"  # The path where data is stored
HIGH_DATA_PATH = r"F:\0wzy_Data\train_set\div2k_train_hr_yuv"  # The path where label is stored

LOG_PATH = "./logs/%s/" % EXP_DATA
CKPT_PATH = "./checkpoints/%s/" % EXP_DATA  # Store the trained models
SAMPLE_PATH = "./samples/%s/" % EXP_DATA  # Store result pic

PATCH_SIZE = (64, 64)  # The size of the input image in the convolutional neural network
BATCH_SIZE = 64  # The number of patches extracted from a picture added to the train set

BASE_LR = 1e-3  # Base learning rate
LR_DECAY_RATE = 0.5
LR_DECAY_STEP = 50
MAX_EPOCH = 500

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path
# model_path = r"I:\train_CNN\checkpoints\CRLCv6_I_QP47~55_C2\CRLCv6_I_QP47~55_C2_202103_100_-22.21.ckpt"

if __name__ == '__main__':
    start = time.time()
    train_list = get_train_list(load_file_list(LOW_DATA_PATH), load_file_list(HIGH_DATA_PATH))
    print(len(train_list))

    with tf.name_scope('input_scope'):
        train_input = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
        train_gt = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
        train_output = model(train_input)
        R = tf.reshape(train_output, [64, 64 * 64, tf.shape(train_output)[-1]])

    with tf.name_scope('loss_scope'), tf.device("/gpu:0"):
        A = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(R, perm=[0, 2, 1]), R)),
                                tf.transpose(R, perm=[0, 2, 1])),
                      tf.reshape(tf.subtract(train_gt, train_input), [64, 64 * 64, 1]))
        # set the loss function
        # Original loss function
        loss = tf.reduce_sum(-(tf.matmul(
            tf.matmul(tf.transpose(tf.reshape(tf.subtract(train_gt, train_input), [64, 64 * 64, 1]), perm=[0, 2, 1]),
                      R), A)))

        avg_loss = tf.placeholder('float32')
        tf.summary.scalar("avg_loss", avg_loss)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(BASE_LR, global_step, LR_DECAY_STEP * 1000, LR_DECAY_RATE,
                                               staircase=True)
    tf.summary.scalar("learning rate", learning_rate)

    optimizer_adam = tf.train.AdamOptimizer(learning_rate, 0.9)
    opt_adam = optimizer_adam.minimize(loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=0)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8

    with tf.Session(config=config) as sess:
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        if not os.path.exists(os.path.dirname(CKPT_PATH)):
            os.makedirs(os.path.dirname(CKPT_PATH))
        if not os.path.exists(SAMPLE_PATH):
            os.makedirs(SAMPLE_PATH)

        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

        sess.run(tf.global_variables_initializer())
        last_epoch = 0

        if model_path:
            print("restore model...", model_path)
            saver.restore(sess, model_path)
            last_epoch = int(os.path.basename(model_path).split(".")[0].split("_")[-2])
            print(last_epoch)
        print("prepare_time:", time.time() - start)

        for epoch in range(last_epoch, last_epoch + MAX_EPOCH):
            shuffle(train_list)
            total_g_loss, n_iter = 0, 0

            epoch_time = time.time()
            total_get_data_time, total_network_time = 0, 0
            for idx in range(1000):
                get_data_time = time.time()
                input_data, gt_data = prepare_nn_data(train_list)
                total_get_data_time += (time.time() - get_data_time)
                network_time = time.time()
                feed_dict = {train_input: input_data, train_gt: gt_data}
                _, l, output, g_step = sess.run([opt_adam, loss, train_output, global_step], feed_dict=feed_dict)
                total_network_time += (time.time() - network_time)
                total_g_loss += l
                n_iter += 1
                del input_data, gt_data, output
            lr, summary = sess.run([learning_rate, merged], {avg_loss: total_g_loss / n_iter})

            file_writer.add_summary(summary, epoch)
            tf.logging.warning(
                "Epoch: [%4d/%4d]  time: %4.4f\t loss: %.8f\t lr: %.16f\t total_get_data_time:"
                " %8f\t total_network_time: %8f" % (
                    epoch, last_epoch + MAX_EPOCH, time.time() - epoch_time, total_g_loss / n_iter, lr,
                    total_get_data_time, total_network_time))
            saver.save(sess, os.path.join(CKPT_PATH, "%s_%03d_%.2f.ckpt" % (EXP_DATA, epoch, total_g_loss / n_iter)))
