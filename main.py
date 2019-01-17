import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import random


# Estimator
class Estimator():
    def __init__(self, path):
        self.price_data = None
        self.scaler = MinMaxScaler()
        self.path = path
        self.num_inputs = 1
        self.num_time_steps = 100
        self.num_neurons = 100
        self.num_outputs = 1

        self.learning_rate = 0.001
        self.num_train_iterations = 6000
        self.batch_size = 1

        self.X = tf.placeholder(tf.float32, [None, self.num_time_steps, self.num_inputs])
        self.y = tf.placeholder(tf.float32, [None, self.num_time_steps, self.num_outputs])

        cell = tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.BasicLSTMCell(num_units=self.num_neurons, activation=tf.nn.relu),
            output_size=self.num_outputs)

        self.outputs, states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)

        self.loss = tf.reduce_mean(tf.square(self.outputs - self.y))  # MSE
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver()

    def get_train_data(self, number_of_data=1000):
        data = pd.read_csv(self.path)
        price_data = data.drop(columns=['market_cap', 'total_volume'])

        price_data = price_data.set_index('snapped_at')
        price_data.index = pd.to_datetime(price_data.index)

        train_set = price_data.tail(number_of_data)

        self.scaler = MinMaxScaler()

        train_scaled = self.scaler.fit_transform(train_set)
        self.price_data = price_data

        return train_scaled

    def next_batch(self, training_data, batch_size, steps):
        """
        INPUT: Data, Batch Size, Time Steps per batch
        OUTPUT: A tuple of y time series results. y[:,:-1] and y[:,1:]
        """
        rand_start = np.random.randint(0, len(training_data) - steps)

        y_batch = np.array(training_data[rand_start:rand_start + steps + 1]).reshape(1, steps + 1)

        """
        shape : (1, steps, 1)
        """
        return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)

    def train_network(self, train_set):
        init = tf.global_variables_initializer()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)

            for iteration in range(self.num_train_iterations):

                X_batch, y_batch = self.next_batch(train_set, self.batch_size, self.num_time_steps)
                sess.run(self.train, feed_dict={self.X: X_batch, self.y: y_batch})

                if iteration % 100 == 0:
                    mse = self.loss.eval(feed_dict={self.X: X_batch, self.y: y_batch})
                    print(iteration, "\tMSE:", mse)

            # Save Model for Later
            self.saver.save(sess, "./Models/btc_time_series_model")

    def predict_data_and_draw_graph(self, train_set):
        with tf.Session() as sess:
            # Use your Saver instance to restore your saved rnn time series model
            self.saver.restore(sess, "./Models/btc_time_series_model")

            # Create a numpy array for your generative seed from the last 12 months of the
            # training set data. Hint: Just use tail(12) and then pass it to an np.array
            train_seed = list(train_set[-self.num_time_steps:])

            # Now create a for loop that
            for iteration in range(self.num_time_steps):
                X_batch = np.array(train_seed[-self.num_time_steps:]).reshape(1, self.num_time_steps, 1)
                y_pred = sess.run(self.outputs, feed_dict={self.X: X_batch})
                train_seed.append(y_pred[0, -1, 0])

        results = self.scaler.inverse_transform(np.array(train_seed[self.num_time_steps:]).reshape(self.num_time_steps, 1))

        date_today = datetime.now()
        days = pd.date_range(date_today, date_today + timedelta(self.num_time_steps - 1), freq='D')

        generated_data = pd.DataFrame(columns=['Date', 'Generated'])
        generated_data['Date'] = days
        generated_data['Generated'] = results
        generated_data = generated_data.set_index('Date')
        # print(generated_data['Generated'].count())

        current_price = self.price_data.tail(1).iat[0, 0]

        expected_profit_list = []
        plt.plot_date(x=generated_data.index, y=generated_data['Generated'], fmt='b-')
        plt.title("GENERATED DATA")
        plt.xlabel("data")
        plt.ylabel("price")
        plt.grid(True)
        plt.show()
        return results, current_price


# Q Learning
class Environment():
    def __init__(self, data_size, price, data):
        self.num_of_states = data_size
        self.previous_price  = price
        self.price_list =  data

    def reset(self):
        self.current_state = 0
        return self.current_state

    def step(self, action):
        if action == 1:
            reward = self.price_list[self.current_state] - self.previous_price
        else:
            reward = 0
        self.current_state = self.current_state + 1
        is_end = (self.current_state >= 99)
        return self.current_state, reward, is_end

    def sample_action(self):
        return random.randint(0,1)


estimator = Estimator("./Data/btc-krw-max.csv")
train_set = estimator.get_train_data()
estimator.train_network(train_set)
results , current_price = estimator.predict_data_and_draw_graph(train_set)
num_time_steps = estimator.num_time_steps

tf.reset_default_graph()

env = Environment(num_time_steps, current_price, results.reshape(num_time_steps).tolist())

epsilon = 0.1
epsilon_minimum_value = 0.001
num_of_action = 2
epoch = 1001
hidden_size = 100
batch_size = 50
num_of_states = num_time_steps
discount = 0.9
learning_rate = 0.2

inputs1 = tf.placeholder(shape=[1,num_of_states], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([num_of_states,2],0,0.01))
Q_out = tf.matmul(inputs1,W)
predict = tf.argmax(Q_out,1)

next_Q = tf.placeholder(shape=[1,2],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(next_Q - Q_out))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
update_model = trainer.minimize(loss)

init = tf.global_variables_initializer()

reward_list = []

with tf.Session() as sess:
    sess.run(init)

    s = env.reset()

    while True:
        a, all_Q = sess.run([predict, Q_out], feed_dict={inputs1:np.identity(num_of_states)[s:s+1]})

        if np.random.rand(1) < epsilon:
            a[0] = env.sample_action()

        s1, r, is_end = env.step(a[0])

        reward_list.append(r)

        Q1 = sess.run(Q_out, feed_dict={inputs1:np.identity(100)[s1:(s1+1)]})

        max_Q1 = np.max(Q1)
        target_Q = all_Q
        target_Q[0, a[0]] = r + discount * max_Q1

        _, W1 = sess.run([update_model, W], feed_dict={inputs1:np.identity(num_time_steps)[s:s+1], next_Q:target_Q})
        s = s1

        if is_end:
            break

plt.plot(reward_list)
plt.show()