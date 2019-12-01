import tensorflow as tf

b1 = tf.constant([10, 100], name='const_b1')

# Placeholders can hold tensors of any shape, if shape is not specified

x = tf.placeholder(tf.int32, name='x')
b0 = tf.placeholder(tf.int32, name='b0')

bx = tf.multiply(b1, x, name='bx')
y = tf.add(bx, b0, name='y')
y_ = tf.subtract(x, b0, name='y_')

with tf.Session() as sess:
    print("bx = ", sess.run(bx, feed_dict={x: [3, 33]}))
    print("Final result ", sess.run(y, feed_dict={x: [5, 50], b0: [7,9]}))
    print("Intermediate result", sess.run(fetches=y, feed_dict={bx: [100, 1000], b0: [7, 9]}))
    print("Two results", sess.run(fetches=[y, y_], feed_dict={x: [5, 50], b0: [7,9]}))

writer = tf.summary.FileWriter('./m3_example2', sess.graph)
writer.close()