import tensorflow as tf
import numpy as np

x = tf.constant([100, 200, 300], name='x')
y = tf.constant([1, 2, 3], name='y')



sum_X = tf.reduce_sum(x, name='sum_x')
sum_Y = tf.reduce_prod(y, name='prod_y')

final_div = tf.div(sum_X, sum_Y, name='final_div')
sess = tf.Session()

R = np.array([1, 2, 3])
print(sess.run(tf.rank(R)))

print(sess.run(final_div))

sess.close()
