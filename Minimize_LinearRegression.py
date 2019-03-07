import tensorflow as tf
import matplotlib.pyplot as plt

X = [1,2,3]
Y = [1,2,3]

W = tf.Variable(5.0)

hypothesis = X * W
learning_rate = 0.1

cost = tf.reduce_mean(tf.square(hypothesis - Y ))
'''
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)
'''
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# get gradients :: gvs = optimizer.compute_gradients(cost)
# apply gradients :: apply_gradients = optimizer.apply_gradients(gvs)
train = optimizer.minimize(cost)
sess = tf.Session()

sess.run(tf.global_variables_initializer())


W_val = []
cost_val = []

for step in range( 100 ) :

    print(step, sess.run(cost), sess.run(W))
    sess.run(train)

