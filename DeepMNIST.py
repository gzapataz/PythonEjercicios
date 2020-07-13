# Deep neural network using tensor flow to recognize handwritten digits
# With 2 convolutional layers
# 1 full connected layer
# 1 Output layer

import tensorflow as tf
from pyglet.gl.agl import struct_GDevice
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data



# We use tensorflow helper to pull down data from de MNIST site
# Digits of 28x28 pixels

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Using interactive session make it the default session so we do not need to to pass sess

sess = tf.InteractiveSession()

# x is the placeholder for the 28 x 28 image data

x = tf.placeholder(tf.float32, shape=[None, 784])

# y_ called y bar and it is a 10 elemento vector, containing the predicted probability of each
# digit(0, 9) class. Such as [0.14, 0.8, 0, 0, 0, 0, 0, 0, 0, 0.06]
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Code for Deep Learning
# Change de MNIST input data from a list of values to a 28 Pixel x 28 Pixel X 1 Gray Scale value Cube

x_image = tf.reshape(x, [-1, 28, 28, 1], name="x_image")

#RELU -> Returns 0 is the value is less than 0 and the value otherwise y = 0, x < 0 and y = x, x >= 0

# Define helper funtion to create weights and biases variables, and convolution, and pooling layers
# We are using RELU as the activation function. This must be initialized to a small positive number
# and with some nouise you don't end up going to zero when comparing diff

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return (tf.Variable(initial))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return (tf.Variable(initial))

# Convolution and pooling. We do convolution and then pooling to control overfitting
# Strides indicates how far and in which direction we shift values as we compute new features
# ksize is the kernel size which is the area we are pooling together
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


## Define Layers in the NN

# 1st Convolution Layer
# 32 Features per each 5x5 patch of the image
# 5x5 pixel is the size of the filter
# 1 is one input channel for gray scale. If we had colors we would have 3 channels (R G B)
W_conv1  = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Do convolution of images, add bias and push through RELU Activation
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# Take results and run through max_pool
# Recevies the result from all neurons, combine and reduce the dimnesionality of the network and the sensitivity of the network

h_pool1 = max_pool_2x2(h_conv1)

# 2nd Convolution Layer
# Process the 32 features from convolution layer1 in 5x5 patch. Return 64 features weights and biases
# 64 Features per each 5x5 patch of the image
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# Do convolution of the output of the 1st convolution layer. Pool results
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# Reduces the image to a 7x7 Image
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer
# We are connecting an image of 7x7 with 64 features to 1024 Neurons
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# Connect output of pooling layer 2 as input to full connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32) # get dropout probability as a training input
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
# Represents the ten digists 0-9
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Define model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss mesurement

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_))

# Loss Optimization

step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# What is correct

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# Review Accuracy of the model

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize variables

init = tf.global_variables_initializer()
sess.run(init)

# Train model
import time

# Define number of steps and and how often to display results

num_steps = 3000
display_every = 1000

# Start timer

start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    step.run(feed_dict={x: batch[0], y_:batch[1], keep_prob:0.5})

    # Periodic status display
    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob:1.0})
        end_time = time.time()
        print("Step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy * 100 ))

# Display Summary
# Time to train

end_time = time.time()
print('Total training time for {0} batchs {1:.2f} seconfs'.format(i+1, end_time-start_time ))

# Accuracy on test data
print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}) * 100))

sess.close()