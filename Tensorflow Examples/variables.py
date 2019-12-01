import tensorflow as tf

# A tensor variable with initial value [2.5, 4]
b1 = tf.Variable([2.5, 4], tf.float32, name='var_b1')

# Placeholders can hold tensors of any shape, if shape is not specified
x = tf.placeholder(tf.float32, name='x')
b0 = tf.Variable([5.0, 10.0], tf.float32, name='b0')

y = b1 * x + b0

# Make the session, explicitly initialize variables
# init is another computational node which needs to be executed to initialize variables
init = tf.global_variables_initializer()

bx = tf.multiply(b1, x, name='bx')
y_ = tf.subtract(x, b0, name='y_')

with tf.Session() as sess:
    # to initialize the variables, we will first execute the init node
    sess.run(init)
    print("Final result ", sess.run(y, feed_dict={x: [10, 100]}))


# Initialize single variable, rather than all the variables in the program
# s = b1 * x
# init = tf.variable_initializer([W])
# sess.run(init)

# Assigning a new value to a variable
# result is a computation node just like any other node and can be run using sess.run(result)
number = tf.Variable(2)
multiplier = tf.Variable(4)
result = number.assign(tf.multiply(number, multiplier))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        print("Ans = ", sess.run(result))
        print("New multiplier = ", sess.run(multiplier.assign_add(1)))

writer = tf.summary.FileWriter('./m3_example3', sess.graph)
writer.close()
