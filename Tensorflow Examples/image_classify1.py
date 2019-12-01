import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

def displayDigit(digit):
  plt.imshow(digit.reshape(28, 28), cmap="Greys", interpolation='nearest')

# labels: Array of all the in training dataset
# Indices: Indices of those images which are nearest neighbors 
# from training dataset to our test digit
def getMajorityPredictedLabel(labels, indices):
  predicted_labels = []
  for i in indices:
    predicted_labels.append(labels[i])
  
  predicted_labels = np.array(predicted_labels)
  counts = np.bincount(predicted_labels)
  return np.argmax(counts)


# Specifying batch size = 1000
training_digits, training_labels = mnist.train.next_batch(1000)

# Specifying batch size = 500
test_digits, test_labels = mnist.test.next_batch(200)

displayDigit(training_digits[1])

tf.reset_default_graph()

# We don't know how many images we gonna read; 
# so non is the first dimension of tensor.shape;
# Each image is represented as 1-D array of pixels
training_digits_pl = tf.placeholder("float", [None, 784])

# We compare one image against entire training dataset to predict its label
test_digits_pl = tf.placeholder("float", [784])

l1_distance = tf.abs(tf.subtract(training_digits_pl, test_digits_pl))

pred_knn_l1 = tf.nn.top_k(tf.negative(l1_distance), k=1)

accuracy = 0
init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  for i in range(len(test_digits)):
    _, indices = sess.run(pred_knn_l1, feed_dict={training_digits_pl: training_digits, test_digits_pl: test_digits[i, :]})
    predicted_label = getMajorityPredictedLabel(training_labels, indices)

    print("Test:", i, "Prediction:", predicted_label)
    print("True label:", test_labels[i])

    if predicted_label == test_labels[i]:
      accuracy += 1./len(test_digits)
  
  print(accuracy)