# SimpleMNIST Mod DataSource PY
# Simple Neural Network to clasify HandWritten Digists from MNIST Dataset
# Modify  SimpleMNIST to obtain the data using Keras

import tensorflow as tf
import numpy as np
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

# We use tensorflow helper to pull down data from de MNIST site
# Digits of 28x28 pixels
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# x is the placeholder for the 28 x 28 image data
tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])

# y_ called y bar and it is a 10 elemento vector, containing the predicted probability of each
# digit(0, 9) class. Such as [0.14, 0.8, 0, 0, 0, 0, 0, 0, 0, 0.06]

y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

# Define Weights and parameters
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define our model

y = tf.nn.softmax(tf.matmul(x, W) + b)

# Loss is Cross Entropy

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Each training in gradient descent we want to minimize cross entropy

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initialize global variables

init = tf.global_variables_initializer()

# Create an interactive session that can span multiple code blocks.

sess = tf.compat.v1.Session()

# Perform the initialization of all global variables

sess.run(init)

# Perform 1000 training iterations

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # get 100 random datapoints from de data. batch_xs = image,
    # batch_ys = digit(0 - 9) class
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# Evaluate how well the model did. Do this by comparing the digit with the highest probability in
# actual (y) ande predicted (y_)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print('Test Accuracy: {0}%'.format(test_accuracy * 100.0))

sess.close()

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
