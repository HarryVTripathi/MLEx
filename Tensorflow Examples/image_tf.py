import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import os

image = img.imread(".//plj.jpg")
print(image.shape)

plj = tf.Variable(image, name='plj')
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Transpose of (1, 1) Tensor
    # Original axes indices: [0, 1, 2]; [1, 0, 2] means swapping the 0th and 1th axes
    transpose = tf.transpose(plj, perm=[1, 0, 2])

    # Instead of using transpose which is not image specific and can
    # work pretty much on any tensor we can use
    # tf.image.transpose_image(x)

    result = sess.run((transpose))
    print(result.shape)
    plt.imshow(result)
    plt.show()

print(image.shape)
plt.imshow(image)
plt.show()