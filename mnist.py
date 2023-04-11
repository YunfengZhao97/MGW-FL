import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import os
import cv2
import time
from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Flatten, Dense, Input, add, \
    Activation, GlobalAveragePooling2D
import numpy as  np
import math
import random
import re
import scipy.stats

tf.config.run_functions_eagerly(True)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)
# tf.config.experimental.set_memory_growth(devices[1], True)

tf.random.set_seed(2345)
stack_n = 18
train_num = 50000
batch_size = 256
weight_decay = 0.0005
test_batch_size = 256
warmup_steps = 2
total_steps = 30#30
train_lr_init = 1e-1
train_lr_end = 1e-7
local_epoch = 5 #5
att_epoch = 30 #30
distillate_epoch = 60 #60


# tf.config.experimental_run_functions_eagerly(True)


def get_model_vars(model, attri=None):
    new_models_vars = {}
    if attri == 'small_coarse':
        for i in range(22):
            conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
            batch_layer_name = 'batch_normalization_%d' % i if i > 0 else 'batch_normalization'
            if i < 19:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights())
                new_models_vars[conv_layer_name] = conv_layer_vars
                batch_layer_vars = np.array(model.get_layer(batch_layer_name).get_weights())
                new_models_vars[batch_layer_name] = batch_layer_vars
            else:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights())
                new_models_vars[conv_layer_name] = conv_layer_vars
        dense_layer_name = 'dense'
        dense_layer_vars = np.array(model.get_layer(dense_layer_name).get_weights())
        new_models_vars[dense_layer_name] = dense_layer_vars

    elif (attri == 'big_coarse'):
        for i in range(28):
            conv_layer_name = 'conv2d_%d' % int(22 + i)
            batch_layer_name = 'batch_normalization_%d' % int(19 + i)
            if i < 25:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights())
                new_models_vars[conv_layer_name] = conv_layer_vars
                batch_layer_vars = np.array(model.get_layer(batch_layer_name).get_weights())
                new_models_vars[batch_layer_name] = batch_layer_vars
            else:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights())
                new_models_vars[conv_layer_name] = conv_layer_vars
        dense_layer_name = 'dense_1'
        dense_layer_vars = np.array(model.get_layer(dense_layer_name).get_weights())
        new_models_vars[dense_layer_name] = dense_layer_vars

    elif attri == 'small_middle':
        for i in range(22):
            conv_layer_name = 'conv2d_%d' % int(22 +28+ i)
            batch_layer_name = 'batch_normalization_%d' % int(19 + 25+i)
            if i < 19:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights())
                new_models_vars[conv_layer_name] = conv_layer_vars
                batch_layer_vars = np.array(model.get_layer(batch_layer_name).get_weights())
                new_models_vars[batch_layer_name] = batch_layer_vars
            else:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights())
                new_models_vars[conv_layer_name] = conv_layer_vars
        dense_layer_name = 'dense_2'
        dense_layer_vars = np.array(model.get_layer(dense_layer_name).get_weights())
        new_models_vars[dense_layer_name] = dense_layer_vars

    elif (attri == 'big_middle'):
        for i in range(28):
            conv_layer_name = 'conv2d_%d' % int(22 +28+ 22+i)
            batch_layer_name = 'batch_normalization_%d' % int(19 +25+19+ i)
            if i < 25:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights())
                new_models_vars[conv_layer_name] = conv_layer_vars
                batch_layer_vars = np.array(model.get_layer(batch_layer_name).get_weights())
                new_models_vars[batch_layer_name] = batch_layer_vars
            else:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights())
                new_models_vars[conv_layer_name] = conv_layer_vars
        dense_layer_name = 'dense_3'
        dense_layer_vars = np.array(model.get_layer(dense_layer_name).get_weights())
        new_models_vars[dense_layer_name] = dense_layer_vars

    elif attri == 'small_fine':

        for i in range(22):
            conv_layer_name = 'conv2d_%d' % int(28 + 22 + 28 + 22 + i)
            batch_layer_name = 'batch_normalization_%d' % int(19 + 25 + 19 + 25 + i)
            if i < 19:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights())
                new_models_vars[conv_layer_name] = conv_layer_vars
                batch_layer_vars = np.array(model.get_layer(batch_layer_name).get_weights())
                new_models_vars[batch_layer_name] = batch_layer_vars
            else:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights())
                new_models_vars[conv_layer_name] = conv_layer_vars
        dense_layer_name = 'dense_4'
        dense_layer_vars = np.array(model.get_layer(dense_layer_name).get_weights())
        new_models_vars[dense_layer_name] = dense_layer_vars

    elif attri == 'big_fine':
        for i in range(28):
            conv_layer_name = 'conv2d_%d' % int(28 + 22 + 28 + 22 + 22 + i)
            batch_layer_name = 'batch_normalization_%d' % int(19 + 25 + 19 + 25 +19+ i)
            if i < 25:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights())
                new_models_vars[conv_layer_name] = conv_layer_vars
                batch_layer_vars = np.array(model.get_layer(batch_layer_name).get_weights())
                new_models_vars[batch_layer_name] = batch_layer_vars
            else:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights())
                new_models_vars[conv_layer_name] = conv_layer_vars
        dense_layer_name = 'dense_5'
        dense_layer_vars = np.array(model.get_layer(dense_layer_name).get_weights())
        new_models_vars[dense_layer_name] = dense_layer_vars
    return new_models_vars


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a, b) / (a_norm * b_norm)
    return cos


def JS_divergence(p, q):  # p=np.asarray([0.65,0.25,0.07,0.03])
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


def distance_of_model(current_model, modelvar, model, current_data):
    Js_D = 0
    prediction1 = []
    prediction2 = []
    x_share = current_data[0]
    y_share = current_data[1]
    for key in current_model:
        model.get_layer(key).set_weights(current_model[key])

    test_db = tf.data.Dataset.from_tensor_slices((x_share, y_share))
    test_db = test_db.map(preprocess, num_parallel_calls=2).batch(test_batch_size)
    test_db = test_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    for (x, y) in tqdm(test_db):
        prediction = model(x)
        prediction = tf.nn.softmax(prediction)
        prediction1.append(np.array(prediction))

    for key in modelvar:
        model.get_layer(key).set_weights(modelvar[key])

    test_db = tf.data.Dataset.from_tensor_slices((x_share, y_share))
    test_db = test_db.map(preprocess, num_parallel_calls=2).batch(test_batch_size)
    test_db = test_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    for (x, y) in tqdm(test_db):
        prediction = model(x)
        prediction = tf.nn.softmax(prediction)
        prediction2.append(np.array(prediction))
    for P, Q in zip(prediction1, prediction2):
        for p, q in zip(P, Q):
            Js_D += (1 - JS_divergence(p, q))

    return Js_D


def softmax(inMatrix):
    outMatrix = tf.nn.softmax(inMatrix)
    outMatrix = np.array(outMatrix)
    return outMatrix


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


def compute_sim(current_model, all_model, current_data, model):
    SIM = []
    for j, modelvar in enumerate(all_model):
        sim = distance_of_model(current_model, modelvar, model, current_data)
        SIM.append(sim)
    SIM = np.array(SIM)

    max_value = SIM.max()
    min_value = SIM.min()
    SIM = (SIM - min_value) / (max_value - min_value)
    SIM = SIM.tolist()

    return SIM


def aggregate(all_model, share_data, model):
    all_model_sims = []
    for i, model_var in enumerate(all_model):
        model_sim = compute_sim(model_var, all_model, share_data[i], model)
        all_model_sims.append(model_sim)
    all_model_sims = np.array(all_model_sims)
    all_model_sims = np.squeeze(all_model_sims)

    # all_model_sims=np.sqrt(all_model_sims * np.transpose(all_model_sims))
    newmodelvars = []

    for i, model_var in enumerate(all_model):
        for key in model_var:
            layer = 0
            tem_num = 0
            for j, another_model in enumerate(all_model):
                tem_num += all_model_sims[i, j]
                layer += all_model_sims[i, j] * another_model[key]
            if tem_num > 0:
                layer = layer / tem_num
            else:
                layer = model_var[key]
            model_var[key] = layer
        newmodelvars.append(model_var)
    F = np.sqrt(all_model_sims * np.transpose(all_model_sims))
    return newmodelvars, F


def get_100(x_fine, y_fine, x_test_fine, y_test_fine, finenumber, client_num):
    '''---------------------------------fine_1 data------------------------------------ '''
    dic = {}
    for i in range(20):
        name = '%d' % i
        dic[name] = []
    for i, lable in enumerate(y_20):
        a = y_fine[i]
        name = '%d' % (int(lable))
        dic[name].append(int(a))
    for key in dic:
        dic[key] = list(set(dic[key]))
    fine1 = []
    for i in finenumber:
        fine1 += dic[str(int(i))]
    num1 = []
    share_num1 = []
    for i in fine1:
        num11 = []
        for j, lable in enumerate(y_fine):
            if i == int(lable):
                num11.append(int(j))
        num112 = random.sample(num11, int(len(num11) / client_num))
        share_num11 = random.sample(num112, int(len(num112) / 2))
        num1 += num112
        share_num1 += share_num11
    random.shuffle(num1)
    random.shuffle(share_num1)
    xtrain_fine1 = []
    ytrain_fine1 = []
    x_share1 = []
    y_share1 = []
    for i in num1:
        xtrain_fine1.append(x_fine[i])
        ytrain_fine1.append(y_fine[i])
    xtrain_fine1 = np.array(xtrain_fine1)
    ytrain_fine1 = np.array(ytrain_fine1)
    for i in share_num1:
        x_share1.append(x_fine[i])
        y_share1.append(y_fine[i])
    x_share1 = np.array(x_share1)
    y_share1 = np.array(y_share1)
    '''----------------------------------------------------------------------'''
    num2 = []
    for i in fine1:
        num22 = []
        for j, lable in enumerate(y_test_fine):
            if i == int(lable):
                num22.append(int(j))
        # num22 = random.sample(num22, int(len(num22) / client_num))
        num2 += num22
    random.shuffle(num2)
    xtest_fine1 = []
    ytest_fine1 = []
    for i in num2:
        xtest_fine1.append(x_test_fine[i])
        ytest_fine1.append(y_test_fine[i])
    xtest_fine1 = np.array(xtest_fine1)
    ytest_fine1 = np.array(ytest_fine1)

    xtrain_fine1, ytrain_fine1, xtest_fine1, ytest_fine1 = image_ip(
        xtrain_fine1, ytrain_fine1, xtest_fine1,ytest_fine1, 10)

    x_share1, y_share1,x_share1, y_share1  = image_ip\
        (x_share1, y_share1 , x_share1, y_share1 , 10)
    return (xtrain_fine1, ytrain_fine1, xtest_fine1, ytest_fine1), (x_share1, y_share1)

'''--------------get_coarse_data--------------------------------------------------------------------------'''

def get_20(x_coarse, y_coarse, x_test_coarse, y_test_coarse, coarse_class, client_num):
    num1 = []
    share_num1 = []
    for i in coarse_class:
        num11 = []
        for j, lable in enumerate(y_coarse):
            if i == int(lable):
                num11.append(int(j))
        num112 = random.sample(num11, int(len(num11) / client_num))
        share_num11 = random.sample(num112, int(len(num112) / 2))
        share_num1 += share_num11
        num1 += num112
    random.shuffle(num1)
    random.shuffle(share_num1)
    '''-----------------------------------------------'''
    xtrain_coarse1 = []
    ytrain_coarse1 = []

    for i in num1:
        xtrain_coarse1.append(x_coarse[i])
        ytrain_coarse1.append(y_coarse[i])
    xtrain_coarse1 = np.array(xtrain_coarse1)
    ytrain_coarse1 = np.array(ytrain_coarse1)
    x_share1 = []
    y_share1 = []
    for i in share_num1:
        x_share1.append(x_coarse[i])
        y_share1.append(y_coarse[i])
    x_share1 = np.array(x_share1)
    y_share1 = np.array(y_share1)
    '''--------------------------------------------------'''
    num2 = []
    for i in coarse_class:
        num22 = []
        for j, lable in enumerate(y_test_coarse):
            if i == int(lable):
                num22.append(int(j))
        # num22 = random.sample(num22, int(len(num22) / client_num))
        num2 += num22
    random.shuffle(num2)
    xtest_coarse1 = []
    ytest_coarse1 = []
    for i in num2:
        xtest_coarse1.append(x_test_coarse[i])
        ytest_coarse1.append(y_test_coarse[i])
    xtest_coarse1 = np.array(xtest_coarse1)
    ytest_coarse1 = np.array(ytest_coarse1)

    xtrain_coarse1, ytrain_coarse1, xtest_coarse1, ytest_coarse1 = image_ip\
        (xtrain_coarse1, ytrain_coarse1,xtest_coarse1, ytest_coarse1, 6)
    x_share1, y_share1,x_share1, y_share1  = image_ip\
        (x_share1, y_share1 , x_share1, y_share1 , 6)

    return (xtrain_coarse1, ytrain_coarse1, xtest_coarse1, ytest_coarse1), (x_share1, y_share1)

def get_10(y_20,y_test_20,x_coarse, y_coarse, x_test_coarse, y_test_coarse, coarsenumber, client_num):
    '''---------------------------------10class data------------------------------------ '''
    num1=[]
    share_num1=[]
    for i in coarsenumber:
        num12=[]
        for j in range(len(y_20)):
            if y_20[j] ==i:
                num12.append(int(j))
        num12_ran = random.sample(num12, int(len(num12) / client_num))
        share_num11 = random.sample(num12_ran, int(len(num12_ran) / 2))
        num1 += num12_ran
        share_num1+=share_num11
    random.shuffle(num1)
    random.shuffle(share_num1)
    xtrain_10 = []
    ytrain_10 = []
    x_share10 = []
    y_share10 = []
    for i in num1:
        xtrain_10.append(x_coarse[i])
        ytrain_10.append(y_coarse[i])
    xtrain_10 = np.array(xtrain_10)
    ytrain_10 = np.array(ytrain_10)
    for i in share_num1:
        x_share10.append(x_coarse[i])
        y_share10.append(y_coarse[i])
    x_share10 = np.array(x_share10)
    y_share10 = np.array(y_share10)
    '''----------------------------------------------------------------------'''
    num2 = []
    for i in coarsenumber:
        num21 = []
        for j in range(len(y_test_20)):
            if y_test_20[j] == i:
                num21.append(int(j))
        # num21_ran = random.sample(num21, int(len(num21) / client_num))
        num2 += num21

    random.shuffle(num2)
    xtest_10 = []
    ytest_10 = []
    for i in num2:
        xtest_10.append(x_test_coarse[i])
        ytest_10.append(y_test_coarse[i])
    xtest_10 = np.array(xtest_10)
    ytest_10 = np.array(ytest_10)

    xtrain_10, ytrain_10, xtest_10, ytest_10 = image_ip(
        xtrain_10, ytrain_10, xtest_10,ytest_10, 4)

    x_share10, y_share10,x_share10, y_share10  = image_ip\
        (x_share10, y_share10 , x_share10, y_share10 , 4)
    return (xtrain_10, ytrain_10, xtest_10, ytest_10), (x_share10, y_share10)


def image_ip(train_images, train_labels, test_images, test_labels, num_class1):
    train_labels = tf.keras.utils.to_categorical(train_labels, num_class1)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_class1)

    train_images = np.array(train_images, dtype=np.float32)
    test_images = np.array(test_images, dtype=np.float32)

    # train_images, test_images = color_normalize(train_images, test_images)

    return train_images, train_labels, test_images, test_labels

#
# def color_normalize(train_images, test_images):
#     mean = [np.mean(train_images[:, :, :, i]) for i in range(3)]  # [125.307, 122.95, 113.865]
#     std = [np.std(train_images[:, :, :, i]) for i in range(3)]  # [62.9932, 62.0887, 66.7048]
#     for i in range(3):
#         train_images[:, :, :, i] = (train_images[:, :, :, i] - mean[i]) / (std[i] * 255)
#         test_images[:, :, :, i] = (test_images[:, :, :, i] - mean[i]) / (std[i] * 255)
#     return train_images, test_images


# def images_augment(images):
#     output = []
#     for img in images:
#         img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#         x = np.random.randint(0, 8)
#         y = np.random.randint(0, 8)
#         if np.random.randint(0, 2):
#             img = cv2.flip(img, 1)
#         output.append(img[x: x + 32, y:y + 32, :])
#     return np.ascontiguousarray(output, dtype=np.float32)


def wide_basic(inputs, in_planes, out_planes, stride):
    if stride != 1 or in_planes != out_planes:
        skip_c = tf.keras.layers.Conv2D(out_planes, kernel_size=1, strides=stride, use_bias=True, padding='SAME')(
            inputs)
    else:
        skip_c = inputs

    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(out_planes, kernel_size=3, strides=1, use_bias=True, padding='SAME')(x)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    x = tf.keras.layers.BatchNormalization(scale=True, center=True, )(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(out_planes, kernel_size=3, strides=stride, use_bias=True, padding='SAME')(x)

    # print("skip:", skip_c.shape)
    # print("x:", x.shape)
    x = tf.keras.layers.add([skip_c, x])

    return x


def wide_layer(out, in_planes, out_planes, num_blocks, stride):
    strides = [stride] + [1] * int(num_blocks - 1)
    # print("strides:", strides)
    for strid in strides:
        # print("i:", i)
        out = wide_basic(out, in_planes, out_planes, strid)
        in_planes = out_planes

    return out


def make_resnet_filter(ins, depth=28, widen_factor=10, model_size=10):
    n = (depth - 4) / 6
    k = widen_factor
    nStages = [16, 16 * k, 32 * k, 64 * k]
    x = tf.keras.layers.Conv2D(nStages[0], kernel_size=3, strides=1, use_bias=True, padding='SAME')(ins)
    x = wide_layer(x, nStages[0], nStages[1], n, stride=1)
    x = wide_layer(x, nStages[1], nStages[2], n, stride=2)
    x = wide_layer(x, nStages[2], nStages[3], n, stride=2)
    x = tf.keras.layers.BatchNormalization(scale=True, center=True)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.AvgPool2D([7, 7])(x)
    x = tf.reshape(x, (-1, 640))
    x = tf.keras.layers.Dense(model_size)(x)
    return x


def make_resnet_filter1(ins, depth=28, widen_factor=10, model_size=10):
    n = (depth - 4) / 6
    k = widen_factor
    nStages = [16, 16 * k, 32 * k, 64 * k]
    x = tf.keras.layers.Conv2D(nStages[0], kernel_size=3, strides=1, use_bias=True, padding='SAME')(ins)
    x = wide_layer(x, nStages[0], nStages[1], n, stride=1)
    x = wide_layer(x, nStages[1], nStages[2], n, stride=2)
    x = wide_layer(x, nStages[2], nStages[3], 1, stride=2)
    x = tf.keras.layers.BatchNormalization(scale=True, center=True)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.AvgPool2D([7, 7])(x)        #————————————————————————————————————————————————————————————————————改，cifar8*8
    x = tf.reshape(x, (-1, 640))
    x = tf.keras.layers.Dense(model_size)(x)
    return x


def l2_loss(model, weights=weight_decay):
    variable_list = []
    for v in model.trainable_variables:
        if 'kernel' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * weights


def cross_entropy(y_true, y_pred):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    return tf.reduce_mean(cross_entropy)


def accuracy(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    # accura = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))
    accura_num = tf.reduce_sum(tf.cast(correct_num, dtype=tf.float32))
    return accura_num


@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        prediction = model(x, training=True)
        ce = cross_entropy(y, prediction)
        l2 = l2_loss(model)
        loss = ce + l2
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return ce, prediction


@tf.function
def test_step(model, x, y):
    prediction = model(x, training=False)
    ce = cross_entropy(y, prediction)
    return ce, prediction


def fine_coarse_accuracy(M1, y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    accura = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))

    prediction = np.array(y_pred)
    y = np.array(y_true)
    y = tf.matmul(y, M1)
    y = np.array(y).squeeze()
    y = tf.cast(y, tf.float32)

    b = np.zeros(prediction.shape)
    b[np.arange(len(prediction)), prediction.argmax(1)] = 1
    b = tf.cast(b, tf.float32)
    prediction1 = tf.matmul(b, M1)
    prediction1 = np.array(prediction1).squeeze()
    prediction1 = tf.cast(prediction1, tf.float32)
    fine_coarse_accu = accuracy(y, prediction1)

    return accura, fine_coarse_accu


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def train(model, optimizer, images, labels, lr):
    sum_loss = 0

    # random shuffle
    seed = np.random.randint(0, 65536)
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)
    # images = images_augment(images)
    train_db = tf.data.Dataset.from_tensor_slices((images, labels))
    train_db = train_db.shuffle(10000).map(preprocess, num_parallel_calls=2).batch(batch_size)
    train_db = train_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    optimizer.lr.assign(lr)
    for (x, y) in tqdm(train_db):
        loss, prediction = train_step(model, optimizer, x, y)
        sum_loss += loss


def test(model, images, labels, order, lr):
    sum_loss = 0
    sum_accuracy = 0
    test_db = tf.data.Dataset.from_tensor_slices((images, labels))
    test_db = test_db.map(preprocess, num_parallel_calls=2).batch(test_batch_size)
    test_db = test_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    for (x, y) in tqdm(test_db):
        loss, prediction = test_step(model, x, y)
        sum_loss += loss
        sum_accuracy += accuracy(y, prediction)
    all_numbers = int(len(labels))
    sum_loss = sum_loss / int(all_numbers / test_batch_size)
    print('+++++++++++++++++++++++++++++++++++++>>>>>>>>>>>>>',sum_accuracy,all_numbers)
    sum_accuracy = sum_accuracy / all_numbers
    print('test, loss:%f, accuracy:%f' % (sum_loss, sum_accuracy))
    with open(order + '.txt', 'a')as f:
        f.write(str(sum_accuracy))
        # f.write(str(lr))
        f.write('\n')
        f.close()


def fine_test(M1, model, images, labels, order, lr):
    sum_loss = 0
    sum_accuracy = 0
    sum_f_c_accuracy = 0
    test_db = tf.data.Dataset.from_tensor_slices((images, labels))
    test_db = test_db.map(preprocess, num_parallel_calls=2).batch(test_batch_size)
    test_db = test_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    for (x, y) in tqdm(test_db):
        loss, prediction = test_step(model, x, y)

        acc, f_c_accuracy = fine_coarse_accuracy(M1, y, prediction)
        sum_loss += loss
        sum_accuracy += acc
        sum_f_c_accuracy += f_c_accuracy
    sum_loss = loss / (len(labels) / test_batch_size)
    sum_accuracy = acc / (len(labels))
    sum_f_c_accuracy = f_c_accuracy / (len(labels))
    print('test, loss:%f, accuracy:%f' %
          (sum_loss, sum_accuracy))
    print('test, loss:%f, fc_accuracy:%f' %
          (sum_loss, sum_f_c_accuracy))
    with open(+ order + '.txt', 'a')as f:
        f.write(str(sum_accuracy))
        # f.write(str(lr))
        f.write('\n')
        f.close()
    with open('fc' + order + '.txt', 'a')as f:
        f.write(str(sum_f_c_accuracy))
        # f.write(str(lr))
        f.write('\n')
        f.close()


def client_train(model, data, order, model_vars, global_epoch, gra_attri):
    (train_images, train_labels, test_images, test_labels) = data
    state_accuracy = []
    for key in model_vars:
        model.get_layer(key).set_weights(model_vars[key])
    for p in range(local_epoch):
        lr = learning_rate(global_epoch * local_epoch + p + 1)
        '''------------------------------------------'''
        train(model, optimizer, train_images, train_labels, lr=lr)
        test(model, test_images, test_labels, gra_attri + order, lr=lr)
    with open(gra_attri + order + '.txt', 'r')as f:
        lines = f.readlines()[-local_epoch:]
        for a in lines:
            result = re.search(r'\d+(\.\d+)?', a).group()
            result = float(result)
            state_accuracy.append(result * 100)
        f.close()
    states = np.mean(state_accuracy)
    modelvars = get_model_vars(model, gra_attri)
    model.save_weights('./' + gra_attri + order)
    return modelvars, states


# def client_fine_train(M, model, train_images, train_labels, test_images, test_labels, order, model_vars, global_epoch):
#     M1 = M.astype('float32')
#     state_accuracy = []
#     for key in model_vars:
#         model.get_layer(key).set_weights(model_vars[key])
#     for p in range(local_epoch):
#         lr = learning_rate(global_epoch * local_epoch + p + 1)
#         '''------------------------------------------'''
#         train(model, optimizer, train_images, train_labels, lr=lr)
#         fine_test(M1, model, test_images, test_labels, order, lr=lr)
#     with open('wide_resnet_fc' + order + '.txt', 'r')as f:
#         lines = f.readlines()[-3:]
#         for a in lines:
#             result = re.search(r'\d+(\.\d+)?', a).group()
#             result = float(result)
#             state_accuracy.append(result * 100)
#         f.close()
#
#     states = [state_accuracy]
#     modelvars = get_model_vars(model, 'fine')
#     model.save_weights('fine' + order)
#     return modelvars, states

def low_high_relation(M, model, low_vars, high_share_data, acc, cross_num):
    L = np.zeros([len(high_share_data), len(low_vars)], np.float32)
    Gv = []

    M1 = M.astype('float32')

    for i, data in enumerate(high_share_data):
        for j, coarse_vars in enumerate(low_vars):
            for key in coarse_vars:
                model.get_layer(key).set_weights(coarse_vars[key])
            (images, lables) = data
            test_db = tf.data.Dataset.from_tensor_slices((images, lables))
            test_db = test_db.map(preprocess, num_parallel_calls=2).batch(test_batch_size)
            test_db = test_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            sum_accuracy = 0
            for (x, y) in tqdm(test_db):
                prediction = model(x, training=False)
                y = np.array(y)
                y = tf.matmul(y, M1)
                y = np.array(y).squeeze()
                y = tf.cast(y, tf.float32)
                accu = accuracy(y, prediction)

                sum_accuracy += accu
            sum_accuracy = sum_accuracy / int(len(lables))

            L[i, j] = sum_accuracy * 100
        threshold = acc[i]
        a = np.argmax(L[i])
        # b = np.max(L[i])
        b = []
        weight = np.flatnonzero(L[i] >= threshold).tolist()
        if weight:
            b.append(int(a))
            Gv.append(b)
        else:
            Gv.append([])
    data1 = pd.DataFrame(L)
    data1.to_csv('low_high_' + 'L' + str(cross_num) + '.csv', mode='a', header=False, index=False)
    data1 = pd.DataFrame(Gv)
    data1.to_csv('low_high_' + 'Gv' + str(cross_num) + '.csv', mode='a', header=False, index=False)
    return Gv


def cross_relationship(M, model, model_vars0, share_data, acc, cross_num):
    L = np.zeros([len(share_data), len(model_vars0)], np.float32)
    Gv = []

    M1 = M.astype('float32')
    # M2=M[1].astype('float32')
    for i, data in enumerate(share_data):
        for j, coarse_vars in enumerate(model_vars0):
            for key in coarse_vars:
                model.get_layer(key).set_weights(coarse_vars[key])
            (images, lables) = data
            test_db = tf.data.Dataset.from_tensor_slices((images, lables))
            test_db = test_db.map(preprocess, num_parallel_calls=2).batch(test_batch_size)
            test_db = test_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            sum_accuracy = 0
            for (x, y) in tqdm(test_db):
                prediction = model(x, training=False)
                prediction = np.array(prediction)
                b = np.zeros(prediction.shape)
                b[np.arange(len(prediction)), prediction.argmax(1)] = 1
                b = tf.cast(b, tf.float32)
                prediction1 = tf.matmul(b, M1)
                prediction1 = np.array(prediction1).squeeze()
                prediction1 = tf.cast(prediction1, tf.float32)
                accu = accuracy(y, prediction1)
                sum_accuracy += accu
            sum_accuracy = sum_accuracy / int(len(lables))

            L[i, j] = sum_accuracy * 100
        threshold = acc[i]
        a = np.argmax(L[i])
        # b = np.max(L[i])
        b = []
        weight = np.flatnonzero(L[i] >= threshold).tolist()

        if weight:
            b.append(int(a))
            Gv.append(b)
        else:
            Gv.append([])
    data1 = pd.DataFrame(L)
    data1.to_csv('L' + str(cross_num) + '.csv', mode='a', header=False, index=False)
    data1 = pd.DataFrame(Gv)
    data1.to_csv('Gv' + str(cross_num) + '.csv', mode='a', header=False, index=False)
    return Gv


def mapp(first_y, second_y, first_dim, second_dim, attri=None):
    input = tf.keras.Input([first_dim])
    output = tf.keras.layers.Dense(second_dim, use_bias=False)(input)
    model = tf.keras.Model(input, output)
    fine_labels = tf.keras.utils.to_categorical(first_y, first_dim)
    coarse_labels = tf.keras.utils.to_categorical(second_y, second_dim)
    train_db = tf.data.Dataset.from_tensor_slices((fine_labels, coarse_labels))
    train_db = train_db.shuffle(10000).map(preprocess, num_parallel_calls=2).batch(128)
    train_db = train_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    optimizer.lr.assign(0.01)
    for i in range(10):
        sum_accuracy = 0
        for (x, y) in tqdm(train_db):
            with tf.GradientTape() as tape:
                pre = model(x)
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, pre))
                correct_num = tf.equal(tf.argmax(y, -1), tf.argmax(pre, -1))
                accuracy = tf.reduce_sum(tf.cast(correct_num, dtype=tf.float32))
                sum_accuracy += accuracy
                gradients = tape.gradient(cross_entropy, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(sum_accuracy / int(len(fine_labels)))
    if attri == '100-20':
        dense_layer_name = 'dense_6'
    elif attri == '100-10':
        dense_layer_name = 'dense_7'
    elif attri == '20-10':
        dense_layer_name = 'dense_8'
    else:
        dense_layer_name = 'dense_9'
    dense_layer_vars = np.array(model.get_layer(dense_layer_name).get_weights())
    # dense_layer_vars = model.get_layer(dense_layer_name).get_weights()
    del model
    data1 = pd.DataFrame(np.squeeze(dense_layer_vars))
    data1.to_csv(attri + '.csv', mode='a', header=False, index=False)
    return dense_layer_vars


def distillate(student_vars, teacher_vars, Gv, share_data, stu_gra, tea_gra):
    new_student_vars = []

    if stu_gra == 'small_coarse':
        stu_output = small_coarse_output
    elif (stu_gra == 'big_coarse'):
        stu_output = big_coarse_output
    elif (stu_gra == 'small_middle'):
        stu_output = small_middle_output
    elif (stu_gra == 'big_middle'):
        stu_output = big_middle_output
    elif (stu_gra == 'small_fine'):
        stu_output = small_fine_output
    else:
        stu_output = big_fine_output

    if tea_gra == 'small_coarse':
        tea_output = small_coarse_output
    elif (tea_gra == 'big_coarse'):
        tea_output = big_coarse_output
    elif (tea_gra == 'small_middle'):
        tea_output = small_middle_output
    elif (tea_gra == 'big_middle'):
        tea_output = big_middle_output
    elif (tea_gra == 'small_fine'):
        tea_output = small_fine_output
    else:
        tea_output = big_fine_output

    for i, client_model_vars in enumerate(student_vars):

        if Gv[i]:
            tea_model = models.Model(img_input, tea_output)
            sub_tea_model = tf.keras.models.Model(inputs=tea_model.input, outputs=tea_model.layers[-2].output)
            for layer in sub_tea_model.layers:
                if layer.name in client_model_vars:
                    sub_tea_model.get_layer(layer.name).set_weights(client_model_vars[layer.name])
            del tea_model

            for num in Gv[i]:

                optimizer.lr.assign(0.1)
                stu_model = models.Model(img_input, stu_output)
                sub_stu_model = tf.keras.models.Model(inputs=stu_model.input,
                                                      outputs=stu_model.layers[-2].output)
                for layer in sub_stu_model.layers:
                    if layer.name in teacher_vars[num]:
                        sub_stu_model.get_layer(layer.name).set_weights(teacher_vars[num][layer.name])
                del stu_model

                data = (share_data[i][0], share_data[i][1])
                for j in range(distillate_epoch):
                    train_db = tf.data.Dataset.from_tensor_slices(data)
                    train_db = train_db.shuffle(1000).map(preprocess, num_parallel_calls=2).batch(batch_size)
                    train_db = train_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

                    for (x, y) in tqdm(train_db):
                        '''--------------------------------------------------------------------------------'''
                        stu_y = sub_tea_model(x)
                        with tf.GradientTape() as tape:
                            pre = sub_stu_model(x, training=True)
                            loss = tf.reduce_mean(
                                tf.losses.MSE(stu_y, pre))
                            gradients = tape.gradient(loss, sub_stu_model.trainable_variables)
                            optimizer.apply_gradients(zip(gradients, sub_stu_model.trainable_variables))
                del sub_tea_model

            stu_model = models.Model(img_input, stu_output)
            for layer in sub_stu_model.layers:
                if layer.name in client_model_vars:
                    layer_vars = np.array(sub_stu_model.get_layer(layer.name).get_weights(), dtype=object)
                    stu_model.get_layer(layer.name).set_weights(layer_vars)
            del sub_stu_model
            for v in stu_model.layers[-1:]:
                v.trainable = True
            for v in stu_model.layers[:-1]:
                v.trainable = False
            optimizer.lr.assign(0.01)
            for j in range(5):
                train_db = tf.data.Dataset.from_tensor_slices(data)
                train_db = train_db.shuffle(1000).map(preprocess, num_parallel_calls=2).batch(batch_size)
                train_db = train_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                for (x, y) in tqdm(train_db):
                    with tf.GradientTape() as tape:
                        prediction = stu_model(x, training=True)
                        ce = cross_entropy(y, prediction)
                        l2 = l2_loss(stu_model)
                        loss = ce + l2
                        gradients = tape.gradient(loss, stu_model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, stu_model.trainable_variables))
            client_model_vars = get_model_vars(stu_model, stu_gra)
            new_student_vars.append(client_model_vars)
            for v in stu_model.layers[:-1]:
                v.trainable = True
            del stu_model
        else:
            new_student_vars.append(client_model_vars)
            continue

    return new_student_vars


rate = math.pow(train_lr_end / train_lr_init, 1 / (total_steps * local_epoch - warmup_steps * local_epoch))


def learning_rate(epoch):
    if epoch < warmup_steps * local_epoch:
        lr = epoch / (warmup_steps * local_epoch) * train_lr_init
    else:
        # lr = train_lr_end + 0.5 * (train_lr_init - train_lr_end) * (
        #     (1 + tf.cos((epoch - warmup_steps*5) / (total_steps*5 - warmup_steps*5) * np.pi))
        # )
        lr = (train_lr_init / (math.pow(rate, warmup_steps * local_epoch))) * (rate ** epoch)
        # lr=0.167*(0.95**epoch)
    return lr


def client_train_one_gra(model, vars, data, global_epoch, gra_attri):
    var1, Acc1 = \
        client_train(model, data[0], order='1', model_vars=vars[0], global_epoch=global_epoch,
                     gra_attri=gra_attri)

    var2, Acc2 = \
        client_train(model, data[1], order='2', model_vars=vars[1], global_epoch=global_epoch,
                     gra_attri=gra_attri)

    var3, Acc3 = \
        client_train(model, data[2], order='3', model_vars=vars[2], global_epoch=global_epoch,
                     gra_attri=gra_attri)

    var4, Acc4 = \
        client_train(model, data[3], order='4', model_vars=vars[3], global_epoch=global_epoch,
                     gra_attri=gra_attri)

    var5, Acc5 = \
        client_train(model, data[4], order='5', model_vars=vars[4], global_epoch=global_epoch,
                     gra_attri=gra_attri)
    return [var1, var2, var3, var4, var5], [Acc1, Acc2, Acc3, Acc4, Acc5]


def load_model(tar_gra):
    new_vars = []
    if tar_gra == 'small_fine':
        small_coarse_output = make_resnet_filter1(img_input, model_size=10)
        small_coarse_model = models.Model(img_input, small_coarse_output)

        sub_model = tf.keras.models.Model(inputs=small_coarse_model.input,
                                          outputs=small_coarse_model.layers[-2].output)
        del small_coarse_model
        small_fine_output = make_resnet_filter1(img_input, model_size=100)
        small_fine_model = models.Model(img_input, small_fine_output)

        for i in range(1, 6):
            sub_model.load_weights('./small_coarse' + str(i))
            sub_model.save_weights('./low-high')
            small_fine_model.load_weights('./low-high')
            modelvars = get_model_vars(small_fine_model, tar_gra)
            new_vars.append(modelvars)

    elif tar_gra == 'small_middle':
        small_coarse_output = make_resnet_filter1(img_input, model_size=10)
        small_coarse_model = models.Model(img_input, small_coarse_output)

        sub_model = tf.keras.models.Model(inputs=small_coarse_model.input,
                                          outputs=small_coarse_model.layers[-2].output)
        del small_coarse_model
        small_middle_output = make_resnet_filter1(img_input, model_size=20)
        small_middle_model = models.Model(img_input, small_middle_output)

        for i in range(1, 6):
            sub_model.load_weights('./small_coarse' + str(i))
            sub_model.save_weights('./low-high')
            small_middle_model.load_weights('./low-high')
            modelvars = get_model_vars(small_middle_model, tar_gra)
            new_vars.append(modelvars)

    elif tar_gra == 'big_fine':
        big_coarse_output = make_resnet_filter(img_input, model_size=10)
        big_coarse_model = models.Model(img_input, big_coarse_output)

        sub_model = tf.keras.models.Model(inputs=big_coarse_model.input,
                                          outputs=big_coarse_model.layers[-2].output)
        del big_coarse_model
        big_fine_output = make_resnet_filter(img_input, model_size=20)
        big_fine_model = models.Model(img_input, big_fine_output)

        for i in range(1, 6):
            sub_model.load_weights('./big_coarse' + str(i))
            sub_model.save_weights('./low-high')
            big_fine_model.load_weights('./low-high')
            modelvars = get_model_vars(big_fine_model, tar_gra)
            new_vars.append(modelvars)

    else:
        big_coarse_output = make_resnet_filter(img_input, model_size=10)
        big_coarse_model = models.Model(img_input, big_coarse_output)

        sub_model = tf.keras.models.Model(inputs=big_coarse_model.input,
                                          outputs=big_coarse_model.layers[-2].output)
        del big_coarse_model
        big_middle_output = make_resnet_filter(img_input, model_size=20)
        big_middle_model = models.Model(img_input, big_middle_output)

        for i in range(1, 6):
            sub_model.load_weights('./big_coarse' + str(i))
            sub_model.save_weights('./low-high')
            big_middle_model.load_weights('./low-high')
            modelvars = get_model_vars(big_middle_model, tar_gra)
            new_vars.append(modelvars)
    return new_vars

def get_middle_label(fine_labels):
    res=[]
    for term in fine_labels:
        if term in [0]:label=0
        elif term in [9,5]:label = 1
        elif term in [3,2,8]:label = 2
        elif term in [6]:label = 3
        elif term in [4]:label = 4
        elif term in [1,7]:label = 5
        res.append(label)
    res=np.array(res)
    return res

def get_coarse_label(fine_labels):
    res=[]
    for term in fine_labels:
        if term in [0,9,5]:label=0
        elif term in [3,2,8]:label = 1
        elif term in [6,4]:label = 2
        elif term in [1,7]:label = 3
        res.append(label)
    res=np.array(res)
    return res

if __name__ == '__main__':
    text_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  #----------改数据集


    (x_100, y_100), (x_test_100, y_test_100) = tf.keras.datasets.mnist.load_data()
    y_20,y_test_20=get_middle_label(y_100),get_middle_label(y_test_100)
    y_10,y_test_10=get_coarse_label(y_100),get_coarse_label(y_test_100)
    x_10,x_test_10,x_20,x_test_20=np.copy(x_100),np.copy(x_test_100),np.copy(x_100),np.copy(x_test_100),


    '''---------------------------------------get_client------------------------------------------'''#中粒度----调分布只是这个————————C2
    # -------------------------------------coarse--------------------------------------
    small_coarse_data1, small_coarse_share1 = \
        get_10(y_20, y_test_20, x_10, y_10, x_test_10, y_test_10, [0,2,4], 5)

    small_coarse_data2, small_coarse_share2 = \
        get_10(y_20, y_test_20, x_10, y_10, x_test_10, y_test_10,  [0,2,4], 5)

    small_coarse_data3, small_coarse_share3 = \
        get_10(y_20, y_test_20, x_10, y_10, x_test_10, y_test_10,[0,2,4], 5)
    small_coarse_data4, small_coarse_share4 = \
        get_10(y_20, y_test_20, x_10, y_10, x_test_10, y_test_10,  [1,3,5], 5)
    small_coarse_data5, small_coarse_share5 = \
        get_10(y_20, y_test_20, x_10, y_10, x_test_10, y_test_10, [1,3,5], 5)

    big_coarse_data1, big_coarse_share1 = \
        get_10(y_20, y_test_20, x_10, y_10, x_test_10, y_test_10, [0,2,4], 5)
    big_coarse_data2, big_coarse_share2 = \
        get_10(y_20, y_test_20, x_10, y_10, x_test_10, y_test_10, [0,2,4], 5)
    big_coarse_data3, big_coarse_share3 = \
        get_10(y_20, y_test_20, x_10, y_10, x_test_10, y_test_10, [0,2,4], 5)
    big_coarse_data4, big_coarse_share4 = \
        get_10(y_20, y_test_20, x_10, y_10, x_test_10, y_test_10, [1,3,5],5)
    big_coarse_data5, big_coarse_share5 = \
        get_10(y_20, y_test_20, x_10, y_10, x_test_10, y_test_10, [1,3,5], 5)
    # -------------------------------------middle---------------------------------------


    small_middle_data1, small_middle_share1 = \
        get_20(x_20, y_20, x_test_20, y_test_20, [0,2,4], 5)
    small_middle_data2, small_middle_share2 = \
        get_20(x_20, y_20, x_test_20, y_test_20, [0,2,4], 5)
    small_middle_data3, small_middle_share3 = \
        get_20(x_20, y_20, x_test_20, y_test_20, [0,2,4], 5)
    small_middle_data4, small_middle_share4 = \
        get_20(x_20, y_20, x_test_20, y_test_20, [1,3,5], 5)
    small_middle_data5, small_middle_share5 = \
        get_20(x_20, y_20, x_test_20, y_test_20, [1,3,5], 5)

    big_middle_data1, big_middle_share1 = \
        get_20(x_20, y_20, x_test_20, y_test_20, [0,2,4], 5)
    big_middle_data2, big_middle_share2 = \
        get_20(x_20, y_20, x_test_20, y_test_20, [0,2,4], 5)
    big_middle_data3, big_middle_share3 = \
        get_20(x_20, y_20, x_test_20, y_test_20, [0,2,4], 5)
    big_middle_data4, big_middle_share4 = \
        get_20(x_20, y_20, x_test_20, y_test_20, [1,3,5], 5)
    big_middle_data5, big_middle_share5 = \
        get_20(x_20, y_20, x_test_20, y_test_20, [1,3,5], 5)

    # --------------------------------------fine------------------------------------
    small_fine_data1, small_fine_share1 = \
        get_100( x_100, y_100, x_test_100, y_test_100, [0,2,4], 5)
    small_fine_data2, small_fine_share2 = \
        get_100( x_100, y_100, x_test_100, y_test_100, [0,2,4], 5)
    small_fine_data3, small_fine_share3 = \
        get_100(x_100, y_100, x_test_100, y_test_100, [0,2,4], 5)
    small_fine_data4, small_fine_share4 = \
        get_100( x_100, y_100, x_test_100, y_test_100, [1,3,5], 5)
    small_fine_data5, small_fine_share5 = \
        get_100( x_100, y_100, x_test_100, y_test_100, [1,3,5], 5)

    big_fine_data1, big_fine_share1 = \
        get_100(x_100, y_100, x_test_100, y_test_100, [0,2,4], 5)
    big_fine_data2, big_fine_share2 = \
        get_100( x_100, y_100, x_test_100, y_test_100, [0,2,4], 5)
    big_fine_data3, big_fine_share3 = \
        get_100(x_100, y_100, x_test_100, y_test_100, [0,2,4], 5)
    big_fine_data4, big_fine_share4 = \
        get_100( x_100, y_100, x_test_100, y_test_100, [1,3,5], 5)
    big_fine_data5, big_fine_share5 = \
        get_100( x_100, y_100, x_test_100, y_test_100, [1,3,5], 5)
    '''-------------------------------model_initialize---------------------------------------------------'''
    small_coarse_data = [small_coarse_data1, small_coarse_data2, small_coarse_data3, small_coarse_data4,
                         small_coarse_data5]
    big_coarse_data = [big_coarse_data1, big_coarse_data2, big_coarse_data3, big_coarse_data4,
                       big_coarse_data5]
    small_middle_data = [small_middle_data1, small_middle_data2, small_middle_data3, small_middle_data4,
                         small_middle_data5]
    big_middle_data = [big_middle_data1, big_middle_data2, big_middle_data3, big_middle_data4,
                       big_middle_data5]
    small_fine_data = [small_fine_data1, small_fine_data2, small_fine_data3, small_fine_data4,
                       small_fine_data5]
    big_fine_data = [big_fine_data1, big_fine_data2, big_fine_data3, big_fine_data4,
                     big_fine_data5]

    small_coarse_share_data = [small_coarse_share1, small_coarse_share2, small_coarse_share3, small_coarse_share4,
                               small_coarse_share5]
    big_coarse_share_data = [big_coarse_share1, big_coarse_share2, big_coarse_share3, big_coarse_share4,
                             big_coarse_share5]
    small_middle_share_data = [small_middle_share1, small_middle_share2, small_middle_share3, small_middle_share4,
                               small_middle_share5]
    big_middle_share_data = [big_middle_share1, big_middle_share2, big_middle_share3, big_middle_share4,
                             big_middle_share5]
    small_fine_share_data = [small_fine_share1, small_fine_share2, small_fine_share3, small_fine_share4,
                             small_fine_share5]
    big_fine_share_data = [big_fine_share1, big_fine_share2, big_fine_share3, big_fine_share4,
                           big_fine_share5]

    img_input = Input(shape=(28, 28, 1))    #——————————————————————————————————————————————————————————————————————————————————改
    '''-------------------------------model_initialize---------------------------------------------------'''
    small_coarse_output = make_resnet_filter1(img_input, model_size=4)
    small_coarse_model = models.Model(img_input, small_coarse_output)
    small_coarse_model.summary()
    small_coarse_var = get_model_vars(small_coarse_model, attri='small_coarse')

    del small_coarse_model

    big_coarse_output = make_resnet_filter(img_input, model_size=4)
    big_coarse_model = models.Model(img_input, big_coarse_output)
    big_coarse_model.summary()
    big_coarse_var = get_model_vars(big_coarse_model, attri='big_coarse')

    del big_coarse_model

    small_middle_output = make_resnet_filter1(img_input, model_size=6)
    small_middle_model = models.Model(img_input, small_middle_output)
    small_middle_var = get_model_vars(small_middle_model, attri='small_middle')

    del small_middle_model

    big_middle_output = make_resnet_filter(img_input, model_size=6)
    big_middle_model = models.Model(img_input, big_middle_output)

    big_middle_var = get_model_vars(big_middle_model, attri='big_middle')

    del big_middle_model

    small_fine_output = make_resnet_filter1(img_input, model_size=10)
    small_fine_model = models.Model(img_input, small_fine_output)

    small_fine_var = get_model_vars(small_fine_model, attri='small_fine')

    del small_fine_model

    big_fine_output = make_resnet_filter(img_input, model_size=10)
    big_fine_model = models.Model(img_input, big_fine_output)

    big_fine_var = get_model_vars(big_fine_model, attri='big_fine')

    del big_fine_model

    optimizer = optimizers.SGD(momentum=0.9, nesterov=True)  #_)-------------------------转化！
    # fine_middle = mapp(y_100, y_20, 10, 6, attri='100-20')
    # fine_coarse = mapp(y_100, y_10, 10, 4, attri='100-10')
    # middle_coarse= mapp(y_20, y_10, 6, 4, attri='20-10')
    # fine_fine = mapp(y_100, y_100, 10, 10, attri='100-100')

    fine_middle = pd.read_csv('100-20.csv', header=None)
    fine_coarse = pd.read_csv('100-10.csv', header=None)
    middle_coarse = pd.read_csv('20-10.csv', header=None)
    fine_fine = pd.read_csv('100-100.csv', header=None)


    small_coarse_vars = [small_coarse_var, small_coarse_var, small_coarse_var, small_coarse_var, small_coarse_var]
    big_coarse_vars = [big_coarse_var, big_coarse_var, big_coarse_var, big_coarse_var, big_coarse_var]
    small_middle_vars = [small_middle_var, small_middle_var, small_middle_var, small_middle_var, small_middle_var]
    big_middle_vars = [big_middle_var, big_middle_var, big_middle_var, big_middle_var, big_middle_var]
    small_fine_vars = [small_fine_var, small_fine_var, small_fine_var, small_fine_var, small_fine_var]
    big_fine_vars = [big_fine_var, big_fine_var, big_fine_var, big_fine_var, big_fine_var]

    for global_epoch in range(total_steps):
        small_coarse_model = models.Model(img_input, small_coarse_output)
        small_coarse_vars, small_coarse_acc = client_train_one_gra(
            small_coarse_model, small_coarse_vars, small_coarse_data, global_epoch, 'small_coarse')
        del small_coarse_model

        big_coarse_model = models.Model(img_input, big_coarse_output)
        big_coarse_vars, big_coarse_acc = client_train_one_gra(
            big_coarse_model, big_coarse_vars, big_coarse_data, global_epoch, 'big_coarse')
        del big_coarse_model

        small_middle_model = models.Model(img_input, small_middle_output)
        small_middle_vars, small_middle_acc = client_train_one_gra(
            small_middle_model, small_middle_vars, small_middle_data, global_epoch, 'small_middle')
        del small_middle_model

        big_middle_model = models.Model(img_input, big_middle_output)
        big_middle_vars, big_middle_acc = client_train_one_gra(
            big_middle_model, big_middle_vars, big_middle_data, global_epoch, 'big_middle')
        del big_middle_model

        small_fine_model = models.Model(img_input, small_fine_output)
        small_fine_vars, small_fine_acc = client_train_one_gra(
            small_fine_model, small_fine_vars, small_fine_data, global_epoch, 'small_fine')
        del small_fine_model

        big_fine_model = models.Model(img_input, big_fine_output)
        big_fine_vars, big_fine_acc = client_train_one_gra(
            big_fine_model, big_fine_vars, big_fine_data, global_epoch, 'big_fine')
        del big_fine_model
        # acc=[1,1,1,1,1]
        whole_acc = [small_coarse_acc, big_coarse_acc, small_middle_acc, big_middle_acc, small_fine_acc, big_fine_acc]
        # whole_acc=[acc,acc,acc,acc,acc]
        if (global_epoch >= 10) and (global_epoch % 2 == 0):#20
            for i in range(6):
                if i == 0:
                    stu_gra = 'small_coarse'
                    tea_gra = 'small_fine'
                    M = pd.read_csv('100-10.csv', header=None)

                    cur_acc = np.array(whole_acc[i])
                    data1 = pd.DataFrame(cur_acc)
                    data1.to_csv(stu_gra + '_acc' + '.csv', mode='a', header=False, index=False)
                    tea_output = small_fine_output
                    tea_model = models.Model(img_input, tea_output)

                    Gv = cross_relationship(M, tea_model, small_fine_vars, small_coarse_share_data, cur_acc, i)
                    del tea_model
                    small_coarse_vars = distillate(small_coarse_vars, small_fine_vars, Gv, small_coarse_share_data, stu_gra,
                                                   tea_gra)


                elif (i == 1):
                    stu_gra = 'big_coarse'
                    tea_gra = 'big_fine'
                    M = pd.read_csv('100-10.csv', header=None)

                    cur_acc = np.array(whole_acc[i])
                    data1 = pd.DataFrame(cur_acc)
                    data1.to_csv(stu_gra + '_acc' + '.csv', mode='a', header=False, index=False)
                    tea_output = big_fine_output  # -------------need to check---------------
                    tea_model = models.Model(img_input, tea_output)

                    Gv = cross_relationship(M, tea_model, big_fine_vars, big_coarse_share_data, cur_acc, i)
                    del tea_model
                    big_coarse_vars = distillate(big_coarse_vars, big_fine_vars, Gv, big_coarse_share_data, stu_gra,
                                                 tea_gra)

                elif (i == 2):
                    stu_gra = 'small_middle'
                    tea_gra = 'big_fine'
                    M = pd.read_csv('100-20.csv', header=None)
                    cur_acc = np.array(whole_acc[i])
                    data1 = pd.DataFrame(cur_acc)
                    data1.to_csv(stu_gra + '_acc' + '.csv', mode='a', header=False, index=False)
                    tea_output = big_fine_output  # -------------need to check---------------
                    tea_model = models.Model(img_input, tea_output)

                    Gv = cross_relationship(M, tea_model, big_fine_vars, small_middle_share_data, cur_acc, i)
                    del tea_model
                    small_middle_vars = distillate(small_middle_vars, big_fine_vars, Gv, small_middle_share_data, stu_gra,
                                                   tea_gra)

                elif (i == 3):
                    stu_gra = 'big_middle'
                    tea_gra = 'big_fine'
                    M = pd.read_csv('100-20.csv', header=None)
                    cur_acc = np.array(whole_acc[i])
                    data1 = pd.DataFrame(cur_acc)
                    data1.to_csv(stu_gra + '_acc' + '.csv', mode='a', header=False, index=False)
                    tea_output = big_fine_output  # -------------need to check---------------
                    tea_model = models.Model(img_input, tea_output)

                    Gv = cross_relationship(M, tea_model, big_fine_vars, big_middle_share_data, cur_acc, i)
                    del tea_model
                    big_middle_vars = distillate(big_middle_vars, big_fine_vars, Gv, big_middle_share_data, stu_gra,
                                                 tea_gra)
                elif (i == 4):
                    stu_gra = 'small_fine'
                    tea_gra = 'big_fine'
                    M = pd.read_csv('100-100.csv', header=None)
                    cur_acc = np.array(whole_acc[i])
                    data1 = pd.DataFrame(cur_acc)
                    data1.to_csv(stu_gra + '_acc' + '.csv', mode='a', header=False, index=False)
                    tea_output = big_fine_output  # -------------need to check---------------
                    tea_model = models.Model(img_input, tea_output)

                    Gv = cross_relationship(M, tea_model, big_fine_vars, small_fine_share_data, cur_acc, i)
                    del tea_model
                    small_fine_vars = distillate(small_fine_vars, big_fine_vars, Gv, small_fine_share_data, stu_gra,
                                                 tea_gra)
                else:
                    pass

                big_fine_model = models.Model(img_input, big_fine_output)
                big_fine_model, F_big_fine = aggregate(big_fine_vars, big_fine_share_data, big_fine_model)
                del big_fine_model

        elif (global_epoch >= 6 and global_epoch < 7 ):#
            for i in range(1, 5):
                if i == 1:
                    stu_gra = 'small_middle'
                    tea_gra = 'small_coarse'
                    M = pd.read_csv('20-10.csv', header=None)
                    cur_acc = np.array(whole_acc[i+1])  # need to check
                    data1 = pd.DataFrame(cur_acc)
                    # data1.to_csv(stu_gra + '_acc' + '.csv', mode='a', header=False, index=False)
                    tea_output = small_coarse_output
                    tea_model = models.Model(img_input, tea_output)

                    Gv = low_high_relation(M, tea_model, small_coarse_vars, small_middle_share_data, cur_acc, i)
                    del tea_model

                    small_middle_vars = distillate(small_middle_vars, small_coarse_vars, Gv, small_middle_share_data,
                                                   stu_gra,
                                                   tea_gra)

                elif i == 2:
                    stu_gra = 'big_middle'
                    tea_gra = 'big_coarse'
                    M = pd.read_csv('20-10.csv', header=None)

                    cur_acc = np.array(whole_acc[i + 1])  # need to check
                    data1 = pd.DataFrame(cur_acc)
                    # data1.to_csv(stu_gra + '_acc' + '.csv', mode='a', header=False, index=False)
                    tea_output = big_coarse_output
                    tea_model = models.Model(img_input, tea_output)

                    Gv = low_high_relation(M, tea_model, big_coarse_vars, big_middle_share_data, cur_acc, i)
                    del tea_model

                    big_middle_vars = distillate(big_middle_vars, big_coarse_vars, Gv, big_middle_share_data, stu_gra,
                                                 tea_gra)

                elif i == 3:
                    stu_gra = 'small_fine'
                    tea_gra = 'small_coarse'
                    M = pd.read_csv('100-10.csv', header=None)

                    cur_acc = np.array(whole_acc[i ])  # need to check
                    data1 = pd.DataFrame(cur_acc)
                    # data1.to_csv(stu_gra + '_acc' + '.csv', mode='a', header=False, index=False)
                    tea_output = small_coarse_output
                    tea_model = models.Model(img_input, tea_output)

                    Gv = low_high_relation(M, tea_model, small_coarse_vars, small_fine_share_data, cur_acc, i)

                    del tea_model

                    small_fine_vars = distillate(small_fine_vars, small_coarse_vars, Gv, small_fine_share_data, stu_gra,
                                                 tea_gra)

                else:
                    stu_gra = 'big_fine'
                    tea_gra = 'big_coarse'
                    M = pd.read_csv('100-10.csv', header=None)

                    cur_acc = np.array(whole_acc[i ])  # need to check
                    data1 = pd.DataFrame(cur_acc)
                    # data1.to_csv(stu_gra + '_acc' + '.csv', mode='a', header=False, index=False)
                    tea_output = big_coarse_output
                    tea_model = models.Model(img_input, tea_output)

                    Gv = low_high_relation(M, tea_model, big_coarse_vars, big_fine_share_data, cur_acc, i)
                    del tea_model
                    big_fine_vars = distillate(big_fine_vars, big_coarse_vars, Gv, big_fine_share_data, stu_gra,
                                               tea_gra)

        else:
            small_coarse_model = models.Model(img_input, small_coarse_output)
            small_coarse_vars, F_small_coarse = aggregate(small_coarse_vars, small_coarse_share_data, small_coarse_model)
            del small_coarse_model

            big_coarse_model = models.Model(img_input, big_coarse_output)
            big_coarse_vars, F_big_coarse = aggregate(big_coarse_vars, big_coarse_share_data, big_coarse_model)
            del big_coarse_model

            small_middle_model = models.Model(img_input, small_middle_output)
            small_middle_vars, F_small_middle = aggregate(small_middle_vars, small_middle_share_data, small_middle_model)
            del small_middle_model
            big_middle_model = models.Model(img_input, big_middle_output)
            big_middle_vars, F_big_middle = aggregate(big_middle_vars, big_middle_share_data, big_middle_model)
            del big_middle_model

            small_fine_model = models.Model(img_input, small_fine_output)
            small_fine_vars, F_small_fine = aggregate(small_fine_vars, small_fine_share_data, small_fine_model)
            del small_fine_model
            big_fine_model = models.Model(img_input, big_fine_output)
            big_fine_vars, F_big_fine = aggregate(big_fine_vars, big_fine_share_data, big_fine_model)
            del big_fine_model

            data1 = pd.DataFrame(F_small_coarse)
            data1.to_csv('F_small_coarse.csv', mode='a', header=False, index=False)
            data1 = pd.DataFrame(F_big_middle)
            data1.to_csv('F_big_middle.csv', mode='a', header=False, index=False)
            print('--------------------------------------------------------------------------')