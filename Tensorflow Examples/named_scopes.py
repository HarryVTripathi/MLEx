import tensorflow as tf

A = tf.constant([3], tf.int32, name='A')
B = tf.constant([5], tf.int32, name='B')
C = tf.constant([6], tf.int32, name='C')

x = tf.placeholder(tf.int32, name='x')

with tf.name_scope("Equation_1"):
    Ax2_1 = tf.multiply(A, tf.pow(x, 2), name="Ax2_1")
    Bx = tf.multiply(B, x, name="Bx2")
    y1 = tf.add_n([Ax2_1, Bx, C], name="y1")

with tf.name_scope("Equation_2"):
    # We can use the same name Ax2_1 instead of Ax2_2 an it won't matter, because it's a different scope
    Ax2_2 = tf.multiply(A, tf.pow(x, 2), name="Ax2_2")
    Bx2 = tf.multiply(B, tf.pow(x, 2), name="Bx2")
    y2 = tf.add_n([Ax2_2, Bx2], name="y2")

with tf.name_scope("Final Sum"):
    y = y1 + y2

with tf.Session as sess:
    print("Final sum", sess.run(y, feed_dict={x: [10]}))

writer = tf.summary.FileWriter('./m3_example4', sess.graph)
writer.close()