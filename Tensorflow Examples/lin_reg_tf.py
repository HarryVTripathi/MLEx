import tensorflow as tf

x = tf.placeholder(tf.int32, shape=[3], name='x')
y = tf.placeholder(tf.int32, shape=[3], name='y')

# Placeholders => A promise of tensor to be provided at runtime
# Now we treat the placeholder like we treated constants throughout the rest of the programs

sum_x = tf.reduce_sum(x, name='sum_x')
prod_y = tf.reduce_prod(y, name='prod_y')

div = tf.div(sum_x, prod_y, name='final_div')

sess = tf.Session()

print('sum_x: ', sess.run(sum_x, feed_dict={x: [100, 200, 300]}))
print('prod_y: ', sess.run(prod_y, feed_dict={y: [1, 2, 3]}))

writer = tf.summary.FileWriter('./m3_example1', sess.graph)
writer.close()

sess.close()
