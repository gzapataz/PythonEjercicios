import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

np.random.seed(42)
house_price = house_size * 100 + np.random.randint(low=20000, high=70000, size=num_house)
plt.plot(house_size, house_price, "bx")
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()

#Funcion que normaliza los valores
def normalize(array):
    return (array - array.mean()) / array.std()

#define el numero de ejemplos de entrenamiento 70% de la data se puede tomar el primer 70%
#Dado que los numeros fueron aleatoreos

num_train_samples = math.floor(num_house * 0.7)

### 1) PREPARACION DE LOS DATOS TRAINING & TESTING
#Definir Train Data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

#Definir Test Data
test_house_size = np.asarray(house_size[num_train_samples:])
test_price = np.asanyarray(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_price_norm = normalize(test_price)

### INFERENCIA

# Set up the TensorFlow placeholders that get updated as we descend down the gradient
# Cada vez que computamos valores por el gradiente descendente pasamos los valores via
# los placeholders para pasarlos al algoritmo

tf_house_size = tf.placeholder("float", name="house_size")
tf_house_price = tf.placeholder("float", name="price")

# Define de variable holding the size_factor and price we set during training.
# We initialize them to some random value based on the normal distribution
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# 2) Define de operation for the predicting values
# Add and Multiply are Tensor flow methods to ensure that the operation is done in the graph

tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# 3) Define the loss function (How Much Error) -> Mean Squared error or MSE

tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_house_price, 2)) / (2 * num_train_samples)

#Optimizer learning rate. The size of the step down the gradient

learning_rate = 0.1

# 4) Define gradient descent optimization that will minimize the loss defined in the operation "cost".

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

#Fin de definicion de variables e inicio de la ejecucion para el entrenamiento
# Initializing Variables

init = tf.global_variables_initializer()

# Launch the graph in the session

with tf.Session() as sess:
    sess.run(init)

    # Set how ofter to display training progress and number of trainin iterations
    display_every = 2
    num_training_iter = 50

    # Calculate the number of lines to animation
    fit_num_plots = math.floor(num_training_iter / display_every)
    # Add storage of factors and offset values from each epoch

    fit_size_factor = np.zeros(fit_num_plots)
    fit_price_offset = np.zeros(fit_num_plots)
    fit_plot_idx = 0

    # Keep iterating the training data
    for iteration in range(num_training_iter):

        # Fit all training data

        for(x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_house_price: y})

        # Display Current Status

        if (iteration + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_house_price:train_price_norm})
            print('Iteration #:', '%04d' % (iteration + 1), "Cost=", '{:.9f}'.format(c), \
                  'size_factor=', sess.run(tf_size_factor), "price_offset", sess.run(tf_price_offset))

            # Save the fit size_factor and price offset to allow animation of learning process
            fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)
            fit_price_offset[fit_plot_idx] = sess.run(tf_price_offset)
            fit_plot_idx = fit_plot_idx + 1

    print('Optimization Finished!')
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_house_price: train_price_norm})
    print('training cost=', training_cost, 'size factor=', sess.run(tf_size_factor), "prices offset=", sess.run(tf_price_offset), '\n')

    # Plot of training and test data and learned regression
    # Get values used to normilized data so we can denormilize data back to its original scale

    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_price, 'go', label='Training data')
    plt.plot(test_house_size, test_price, 'mo', label='Testing data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
             (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,\
            label='Learned Regression')
    plt.legend(loc='upper left')
    plt.show()

    # Plot another graph that animation of the gradient descent secuentially adjusted size_factor and price Offset to
    # find the values returned the best fit line

    fig, ax = plt.subplots()
    line, = ax.plot(house_size, house_price)

    plt.rcParams["figure.figsize"] = (10, 8)
    plt.title('Gradient Descent Fitting Regression Line')
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_price, 'go', label='Training data')
    plt.plot(test_house_size, test_price, 'mo', label='Testing data')

    def animate(i):
        line.set_xdata(train_house_size_norm * train_house_size_std + train_house_size_mean) #update the data
        line.set_ydata((fit_size_factor[i] * train_house_size_norm + fit_price_offset[i]) * train_price_std + train_price_mean)
        return line,

    # Init only requied for blitting to give a clean state

    def initAnim():
        line.set_ydata(np.zeros(shape=house_price.shape[0])) #set y's to 0
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, fit_plot_idx), init_func=initAnim,
                                  interval=1000, blit=True)
    plt.show()

