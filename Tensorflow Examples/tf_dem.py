import tensorflow as tf
a = tf.constant(6, name='const_a')
b = tf.constant(6, name='const_b')
c = tf.constant(6, name='const_c')
d = tf.constant(6, name='const_d')
mul = tf.multiply(a, b, name='null')
div = tf.div(c, d, name='div')
addn = tf.add_n([mul, div], name='addn')
print(addn)
# Tensor("addn:0", shape=(), dtype=int32)
sess = tf.Session()
# 2019-11-03 21:23:18.317992: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
sess.run(addn)
writer = tf.summary.FileWriter('/m2_example1', sess.graph)
writer.close()
sess.close()

# run in command line:
# tensorboard --logdir="new_dir"