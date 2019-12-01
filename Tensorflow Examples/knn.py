import tensorflow as tf
import numpy as np

# Import MNIST dataset

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

# Specifying batch size = 5000
training_digits, training_labels = mnist.train.next_batch(5000)

# Specifying batch size = 500
test_digits, test_labels = mnist.test.next_batch(200)

# Placeholders to hold training and test digits
# The first dimension(none) indicates the index of each image
training_digits_pl = tf.placeholder("float", [None, 784])
test_digits_pl = tf.placeholder("float", [784])

# Nearest neighbor calculation using L1 distance 
# Every element in the vector representing training images, 
# is added to its corresponding element in test image 
l1_distance = tf.abs(tf.add(training_digits_pl, tf.negative(test_digits_pl)))

# Sums up all the elements of each vector that we got so far
distance = tf.reduce_sum(l1_distance, axis=1)
print(distance)

# Prediction: Get the minimum distance index
prediction = tf.arg_min(distance, 0)

accuracy = 0

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  print(len(test_digits))
  for i in range(len(test_digits)):
    nn_index = sess.run(prediction, feed_dict={training_digits_pl: training_digits, test_digits_pl: test_digits[i, :]})
    print("Test:", i, "Prediction:", np.argmax(training_labels[nn_index]))
    print("True label:", np.argmax(test_labels[i]))

    if np.argmax(training_labels[nn_index] == test_labels[i]):
      accuracy += 1./len(test_digits)
  
  print("Accuracy:", accuracy)