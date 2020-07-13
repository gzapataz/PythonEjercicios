import tensorflow as tf
print("We're using TF", tf.__version__)

sess = tf.Session()

hello = tf.constant("Hello World")
print(sess.run(hello))

a = tf.constant(30)
b = tf.constant(20)

print('a+b={0}'.format(sess.run(a+b)))
