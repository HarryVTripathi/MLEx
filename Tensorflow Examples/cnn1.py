import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)
training_digits, training_labels = mnist.train.next_batch(1000)
test_digits, test_labels = mnist.test.next_batch(200)

HEIGHT=28
WIDTH=28
CHANNELS=1
NO_INPUT = HEIGHT * WIDTH

CONVOLUTION_FEATURE_MAPS1=32
CONVOLUTION_KERNEL_SIZE1=3
CONVOLUTION_STRIDE1=1
CONVOLUTION_PADDING1="same"

CONVOLUTION_FEATURE_MAPS2=64
CONVOLUTION_KERNEL_SIZE2=3
CONVOLUTION_STRIDE2=2
CONVOLUTION_PADDING2="same"

# Fully connected layer which follows convolutional and pooling layers
NO_FULLY_CONNECTED_LAYERS=64
NO_OUTPUTS=10
NO_EPOCHS=5
BATCH_SIZE=100

pool3_feature_maps=CONVOLUTION_FEATURE_MAPS2

#######################################################################

def displayDigit(digit):
  plt.imshow(digit.reshape(28, 28), cmap="Greys", interpolation='nearest')


tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[None, NO_INPUT], name="X")

# Reshaping data to get a 2D tensor
# -1 tells that first argument continues to be batch size
X_reshaped = tf.reshape(X, shape=[-1, HEIGHT, WIDTH, CHANNELS])

y = tf.placeholder(tf.int32, shape=[None], name="y")

convolutional1 = tf.layers.conv2d(
  X_reshaped,
  filters=CONVOLUTION_FEATURE_MAPS1,
  kernel_size=CONVOLUTION_KERNEL_SIZE1,
  strides=CONVOLUTION_STRIDE1,
  padding=CONVOLUTION_PADDING1,
  activation=tf.nn.relu,
  name="convolutional1"
)

# Output of the first layer is the input to the second layer
convolutional2 = tf.layers.conv2d(
  convolutional1,
  filters=CONVOLUTION_FEATURE_MAPS2,
  kernel_size=CONVOLUTION_KERNEL_SIZE2,
  strides=CONVOLUTION_STRIDE2,
  padding=CONVOLUTION_PADDING2,
  activation=tf.nn.relu,
  name="convolutional2"
)

# To understand how the shape changes
print(convolutional2.shape)

# Kernel size = [batch_size, height, width, channels]
# Stride = [batch_size, height, width, channels]
# Pooling layer connected to the output of second convolutional layer

pool3 = tf.nn.max_pool(
  convolutional2,
  ksize=[1, 2, 2, 1],
  strides=[1, 2, 2, 1],
  padding="VALID"
)

print(pool3.shape)

# In order to feed this tensor to a fully connected neural network, flatten it
# -1 => First dimension is still the batch size
pool3_flat = tf.reshape(pool3, shape=[-1, pool3_feature_maps * 7 * 7])

# Connected to the output of the pooling layer
fully_connected = tf.layers.dense(
  pool3_flat,
  NO_FULLY_CONNECTED_LAYERS,
  activation=tf.nn.relu,
  name="fully_connected"
)

# SoftMax prediction layer. Input of fc layer is fed to it.
logits = tf.layers.dense(fully_connected, NO_OUTPUTS, name="output")

# Takes in a dense layer and applies softmax activation to every layer
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
  logits=logits,
  labels=y
)

loss = tf.reduce_mean(xentropy)

# to minimize the loss
optimizer = tf.train.AdamOptimizer()

training_ops = optimizer.minimize(loss)

# k=1 => The label with the highest probablity matches th actual label
cross_validation=tf.nn.in_top_k(logits, y, 1)

accuracy = tf.reduce_mean(tf.cast(cross_validation, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init)

  for epoch in range(NO_EPOCHS):
    for i in range(mnist.train.num_examples // BATCH_SIZE):
      X_batch, y_batch = mnist.train.next_batch(BATCH_SIZE)
      sess.run(training_ops, feed_dict={X: X_batch, y: y_batch})

    acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
    acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
    
    print(epoch, "Training accuracy", acc_train, "Test accuracy", acc_test)
    save_path = saver.save(sess, "./cnnModel")
