import tensorflow as tf
import numpy as np

x_data = np.random.rand(2, 100).astype("float32")
y_data = np.dot([0.100, 0.200], x_data) + 0.300

W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(W, x_data) + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in xrange(501):
    sess.run(train)
    if step % 50 == 0:
        print step, sess.run(W), sess.run(b)

# Learns best fit is W: [0.1], b: [0.3]
