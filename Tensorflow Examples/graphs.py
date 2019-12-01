import tensorflow as tf

# We can explicitly define as many graphs as we need
# Every tf program has a default graph associated with it
# Every computation falls into this default graph
# We've been implicitly using this graph

# Explicitly instantiating a graph
g1 = tf.Graph()

# If we want our tensors and operators be added to this graph
with g1.as_default():
    with tf.Session as sess:

        y = A * x + b

        # Ensure all of our tensors are in g1
        assert y.graph is g1

g2 = tf.Graph()

with g2.as_default():
    with tf.Session as sess:

        y = A * x + b

        # Ensure all of our tensors are in g1
        assert y.graph is g2

# When graph is not specified explicitly, all our computations are added to the default graph
dg = tf.get_default_graph()
