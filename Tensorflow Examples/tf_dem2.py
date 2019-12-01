import tensorflow as tf

print(tf.__version__)

a = tf.constant(6.5, name='const_a')
b = tf.constant(3.4, name='const_b')
c = tf.constant(3.0, name='const_c')
d = tf.constant(100.2, name='const_d')

sq_a = tf.square(a, name='square_a')
b_pow_c = tf.pow(b, c, name='b_pow_c')
sqrt_d = tf.sqrt(d, name='sqrt_d')

tf.rank(sqrt_d)

final_sum = tf.add_n([sq_a, b_pow_c, sqrt_d], name='final_sum')

sess = tf.Session()
print(sess.run(final_sum))

writer = tf.summary.FileWriter('./m2_example2', sess.graph)
writer.close()
sess.close()