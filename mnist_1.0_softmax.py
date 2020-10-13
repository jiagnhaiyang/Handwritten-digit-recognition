# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflowvisu
import mnistdata
import math
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# 1层10个softmax神经元的神经网络
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: 矩阵为100个灰度，像素值为28×28的图像，将一张图片像素值平铺成一个1×784的向量，所以向量X为100×784
#              b: 10维的偏差向量
#              +: 将向量加到矩阵的每一行上
#              softmax(matrix) 将softmax应用在每一行上
#              softmax(line) 对每个值应用exp，然后除以结果行的范数 applies an exp to each value then divides by the norm of the resulting line
#              Y: 输出矩阵为100行，10列。

# 下载mnist数据集，训练图片有60000张，测试图片有10000张
mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

# input X: 灰度为28x28的图像，第一个维度(无)将对小批处理中的图像进行索引
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# Y_为图片真实的label值
Y_ = tf.placeholder(tf.float32, [None, 10])
# weights W[784, 10]   784=28*28
W = tf.Variable(tf.zeros([784, 10]))
# biases b[10]
b = tf.Variable(tf.zeros([10]))

# 将图像平铺成一行像素
# 形状定义中的-1表示“唯一可能保持元素数量的维度”
XX = tf.reshape(X, [-1, 784])

# 模型
Y = tf.nn.softmax(tf.matmul(XX, W) + b)
"""
Y预测值为   [0.1，0.2，0.1，0.3，0.2，0.1，0.9，0.2，0.1，0.1]
Y_真实值    [0  , 0  , 0  , 0  , 0  , 0  , 1  , 0  , 0  , 0  ]
"""
# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: 通过模型公式计算出的预测值
#                           Y_:标签的真实值

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
#tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # normalized for batches of 100 images,
                                                          # *10 because  "mean" included an unwanted division by 10

# accuracy of the trained model, between 0 (worst) and 1 (best)
#tf.argmax(Y,1)是找到Y向量中最大元素的下标
#tf,equal()比较两个向量最大元素下标是否相等，相等返回True，否则返回False
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
#correct_prediction是一个布尔型数组
#tf.reduce_mean用于计算tensor沿着指定数轴上的平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

# matplotlib visualisation
allweights = tf.reshape(W, [-1])
allbiases = tf.reshape(b, [-1])
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines
datavis = tensorflowvisu.MnistDataVis()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y})
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})


datavis.animate(training_step, iterations=2000+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.


