# -*- coding:utf-8 -*-  
import tensorflow as tf
import numpy as np
import time
import os
import sys
import pickle
import random


class_num = 10
image_size = 32
img_channels = 3
iterations = 200
batch_size = 250
total_epoch = 164
weight_decay = 0.0003
dropout_rate = 0.5
momentum_rate = 0.9
log_save_path = 'vgg_logs'
model_save_path = '/model/'


def download_data():
    dirname = 'cifar10-dataset'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    fname = './CAFIR-10_data/cifar-10-python.tar.gz'
    fpath = './' + dirname

    download = False
    if os.path.exists(fpath) or os.path.isfile(fname):
        download = False
        print("DataSet already exist!")
    else:
        download = True
    if download:
        print('Downloading data from', origin)
        import urllib.request
        import tarfile

        def reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration+1))
            percent = min(int(count*block_size*100/total_size),100)
            sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                            (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()

        urllib.request.urlretrieve(origin, fname, reporthook)
        print('Download finished. Start extract!', origin)
        if fname.endswith("tar.gz"):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()
        elif fname.endswith("tar"):
            tar = tarfile.open(fname, "r:")
            tar.extractall()
            tar.close()


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels


def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels


def prepare_data():
    print("======Loading data======")
    download_data()
    data_dir = './cifar10-dataset'
    image_dim = image_size * image_size * img_channels
    meta = unpickle(data_dir + '/batches.meta')

    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data(train_files, data_dir, label_count)
    test_data, test_labels = load_data(['test_batch'], data_dir, label_count)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(input, k_size=1, stride=1, name=None):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                          padding='SAME', name=name)


def batch_norm(input):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                    nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch


def data_preprocessing(x_train,x_test):

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch


def learning_rate_schedule(epoch_num):
    if epoch_num < 81:
        return 0.1
    elif epoch_num < 121:
        return 0.01
    else:
        return 0.001



if __name__ == '__main__':
    x = tf.placeholder(tf.float32,[None, 120, 120, 3])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network
    
    W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 128], initializer=tf.contrib.keras.initializers.he_normal())
    output1 = tf.nn.relu(batch_norm(conv2d(x, W_conv1_1)))

    W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 128, 128], initializer=tf.contrib.keras.initializers.he_normal())
    output2 = tf.nn.relu(batch_norm(conv2d(output1, W_conv1_2)))
    output3 = max_pool(output2, 2, 2, "pool1")

    W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 128, 256], initializer=tf.contrib.keras.initializers.he_normal())
    output4 = tf.nn.relu(batch_norm(conv2d(output3, W_conv2_1)))

    W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
    output5 = tf.nn.relu(batch_norm(conv2d(output4, W_conv2_2)))
    output6 = max_pool(output5, 2, 2, "pool2")

    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 256, 512], initializer=tf.contrib.keras.initializers.he_normal())
    output7 = tf.nn.relu(batch_norm(conv2d(output6,W_conv3_1)))

    W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    output8 = tf.nn.relu(batch_norm(conv2d(output7, W_conv3_2)))
    output9 = max_pool(output8, 2, 2, "pool3")

    # output = tf.contrib.layers.flatten(output)
    output = tf.reshape(output9, [-1, 4*4*512])

    W_fc1 = tf.get_variable('fc1', shape=[8192, 1024], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc1)))
    output = tf.nn.dropout(output, keep_prob)

    W_fc3 = tf.get_variable('fc3', shape=[1024, 10], initializer=tf.contrib.keras.initializers.he_normal())
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc3)))
    # output  = tf.reshape(output,[-1,10])

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initial an saver to save model
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_save_path,sess.graph)
        saver.restore(sess, 'base7.ckpt')
        testData = np.load('testDataBoat44x120x120.npy')
        testBatch = np.zeros((44,120,120,3))
        testBatch[:, :, :, 0] = (testData[:, :, :] - np.mean(testData[:, :, :])) / np.std(testData[:, :, :])
        testBatch[:, :, :, 1] = testBatch[:, :, :, 0]
        testBatch[:, :, :, 2] = testBatch[:, :, :, 0]
        outputFM  = sess.run([output9],
                                feed_dict={x: testBatch, keep_prob: 1.0, train_flag: False})
        np.save('outputFM9_boat.npy',np.array(outputFM[0]))