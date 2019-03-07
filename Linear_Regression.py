import tensorflow as tf

#X,Y Data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#Our Hypothesis XW+b
Hypothesis = X * W + b

#Cost / Loss Function
#reduce_mean Clac Avarage
Cost = tf.reduce_mean(tf.square(Hypothesis - Y))

#Minimize

Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
Train = Optimizer.minimize(Cost)


# Launch the graph in a session

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    Cost_Val, W_Val, b_Val, _ = sess.run([Cost, W, b, Train], feed_dict={X:[1,2,3,4,5], Y:[2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, Cost_Val, W_Val, b_Val, )

