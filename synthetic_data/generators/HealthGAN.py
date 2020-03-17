"""
HealthGAN with gradient penalties
"""

import os
import sys
import time
import pickle as pkl
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

class HealthGAN():
    """
    Wasserstein GAN with gradient penalties
    """
    params = {
        'base_nodes': 64,
        'critic_iters': 5,  # number of discriminator iterations
        'lambda': 10,  # paramter for gradient penalty
        'num_epochs': 100000  # how long to train for
    }

    def __init__(self,
                 train_file,
                 test_file,
                 critic_iters=None,
                 base_nodes=None,
                 num_epochs=None):

        tf.reset_default_graph()

        # set custom options
        if critic_iters:
            self.params['critic_iters'] = critic_iters

        # Set number of epochs
        if num_epochs:
            self.params['num_epochs'] = num_epochs
            
        # Read the training data file
        train_data = pd.read_csv(train_file)
        self.col_names = train_data.columns
        train_data = train_data.values

        # Read the test data file
        self.test_data = pd.read_csv(test_file).values

        # Extract features from the train_data files
        self.params['n_observations'] = train_data.shape[0]
        self.params['n_features'] = train_data.shape[1]

        # Create 1.5 and 2 times for the generator network dimensions
        self.params['1.5_n_features'] = round(1.5 * self.params['n_features'])
        self.params['2_n_features'] = 2 * self.params['n_features']

        # Create 2 and 4 times for the discriminator
        if base_nodes:
            self.params['base_nodes'] = base_nodes
        self.params['2_base_nodes'] = 2 * self.params['base_nodes']
        self.params['4_base_nodes'] = 4 * self.params['base_nodes']

        # number of observations divided by the number of critic iterations
        # rounded down to the nearest multiple of 100
        self.params['batch_size'] = int(train_data.shape[0] / self.params['critic_iters']) // 100 * 100

        self.train_batcher = self.__data_batcher(train_data, self.params['batch_size'])

        self.__print_settings()

        # Predefine values that will be set later
        self.real_data = None
        self.gen_loss = None
        self.disc_loss = None
        self.gen_train_op = None
        self.disc_train_op = None
        self.rand_noise_samples = None

        # Define lists to store data
        self.disc_loss_all = []
        self.gen_loss_all = []
        self.disc_loss_test_all = []
        self.time_all = []

        # Define the computation graph
        self.__create_graph()


    def __data_batcher(self, data, batch_size):
        """
        Create yield function for given data and batch size
        """

        def get_all_batches():
            """
            Yield function (generator) for all batches in data
            """
            # Shuffle in place
            np.random.shuffle(data)

            # Get total number of evenly divisible batches
            # with shape: (num_batches, batch_size, n_features)
            batches = data[:(data.shape[0] // batch_size) * batch_size]
            batches = batches.reshape(-1, batch_size, data.shape[1])

            # Iterate through all batches and yield them
            for i, _ in enumerate(batches):
                yield np.copy(batches[i])

        def infinite_data_batcher():
            """
            Createa a generator that yields new batches every time it is called
            """
            while True:
                for batch in get_all_batches():
                    yield batch

        return infinite_data_batcher()

    def __print_settings(self):
        """
        Print the settings of the whole system
        """
        for k, v in self.params.items():
            print(f'{k + ":":18}{v}')
        print()

    def __generator(self, inpt):
        """
        Create the generator graph
        """
        # First dense layer
        output = tf.contrib.layers.fully_connected(
            inpt,
            self.params['2_n_features'],
            activation_fn=tf.nn.relu,
            scope='Generator.1',
            reuse=tf.AUTO_REUSE)

        # Second dense layer
        output = tf.contrib.layers.fully_connected(
            output,
            self.params['1.5_n_features'],
            activation_fn=tf.nn.relu,
            scope='Generator.2',
            reuse=tf.AUTO_REUSE)

        # Third dense layer
        output = tf.contrib.layers.fully_connected(
            output,
            self.params['n_features'],
            activation_fn=tf.nn.sigmoid,
            scope='Generator.3',
            reuse=tf.AUTO_REUSE)

        return output

    def __discriminator(self, output):
        """
        Create the discriminator graph
        """
        # Create first dense layer
        output = tf.contrib.layers.fully_connected(
            output,
            self.params['base_nodes'],
            activation_fn=tf.nn.leaky_relu,
            scope='Discriminator.1',
            reuse=tf.AUTO_REUSE)

        # Create second dense layer
        output = tf.contrib.layers.fully_connected(
            output,
            self.params['2_base_nodes'],
            activation_fn=tf.nn.leaky_relu,
            scope='Discriminator.2',
            reuse=tf.AUTO_REUSE)

        # Create third dense layer
        output = tf.contrib.layers.fully_connected(
            output,
            self.params['4_base_nodes'],
            activation_fn=tf.nn.leaky_relu,
            scope='Discriminator.3',
            reuse=tf.AUTO_REUSE)

        # Create fourth dense layer
        output = tf.contrib.layers.fully_connected(
            output,
            1,
            activation_fn=None,
            scope='Discriminator.4',
            reuse=tf.AUTO_REUSE)

        return output

    def __create_graph(self):
        """
        create computation graph
        """

        # Create the placeholder for real data and generator for fake
        self.real_data = tf.placeholder(tf.float32,
                                        shape=[self.params['batch_size'], self.params['n_features']],
                                        name="RealData")

        # Create a noise data set of size of the number of samples by 100
        noise = tf.random_normal([self.params['batch_size'], 100])
        fake_data = self.__generator(noise)

        # Run the discriminator for both types of data
        disc_real = self.__discriminator(self.real_data)
        disc_fake = self.__discriminator(fake_data)

        # Create the loss for generator and discriminator
        self.gen_loss = -tf.reduce_mean(disc_fake)
        self.disc_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        # Add the gradient penalty to discriminator loss
        # and create random split for data
        alpha = tf.random_uniform(shape=[self.params['batch_size'], 1], minval=0, maxval=1)

        # Combine real and fake
        interpolates = (alpha * self.real_data) + ((1 - alpha) * fake_data)

        # Compute gradients of dicriminator values
        gradients = tf.gradients(self.__discriminator(interpolates), [interpolates])[0]

        # Calculate the 2 norm of the gradients
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))

        # Aubtract 1, square, and then use lambda parameter to scale
        gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
        self.disc_loss += self.params['lambda'] * gradient_penalty

        # Use Adam Optimizer on losses
        gen_params = [
            v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if 'Generator' in v.name
        ]

        self.gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
                self.gen_loss, var_list=gen_params)
        disc_params = [
            v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if 'Discriminator' in v.name
        ]
        self.disc_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
                self.disc_loss, var_list=disc_params)

        # For generating samples
        rand_noise = tf.random_normal([100000, 100], name="RandomNoise")
        self.rand_noise_samples = self.__generator(rand_noise)

        with tf.Session() as session:
            _ = tf.summary.FileWriter('./logs_new', session.graph)

    def train(self):
        """
        The method trains the HealthGAN on the data provided.

        Outputs
        -------
        Tensorflow graph:
            The final ouput generated is a tensorflow graph.
        """
        # "saver" object for saving the model
        saver = tf.train.Saver()

        with tf.Session() as session:
            # Initialize variables
            session.run(tf.global_variables_initializer())

            for epoch in range(self.params['num_epochs']):
                start_time = time.time()

                disc_loss_list = []
                for i in range(self.params['critic_iters']):
                    # Get a batch
                    train = next(self.train_batcher)
                    # Run one critic iteration
                    disc_loss, _ = session.run(
                        [self.disc_loss, self.disc_train_op],
                        feed_dict={self.real_data: train})
                    disc_loss_list.append(disc_loss)

                # Run one generator train iteration
                gen_loss, _ = session.run([self.gen_loss, self.gen_train_op])

                # Save the loss and time of iteration
                self.time_all.append(time.time() - start_time)
                self.disc_loss_all.append(disc_loss_list)
                self.gen_loss_all.append(gen_loss)

                # Print the results
                if epoch < 10 or epoch % 100 == 99:
                    print((f'Epoch: {epoch:5} '
                           f'[D loss: {self.disc_loss_all[-1][-1]:7.4f}] '
                           f'[G loss: {self.gen_loss_all[-1]:7.4f}] '
                           f'[Time: {self.time_all[-1]:4.2f}]'))

                # If at epoch ending 999 check test loss
                if epoch == 0 or epoch % 1000 == 999:
                    # Shuffle test in place
                    np.random.shuffle(self.test_data)
                    test_disc_loss = session.run(
                        self.disc_loss,
                        feed_dict={
                            self.real_data:
                            self.test_data[:self.params['batch_size']]
                        })
                    self.disc_loss_test_all.append(test_disc_loss)
                    print(f'Test Epoch: [Test D loss: {self.disc_loss_test_all[-1]:7.4f}]')

                # If at epoch ending 99999 generate large
                if epoch == (self.params['num_epochs'] - 1):
                    for i in range(10):
                        samples = session.run(self.rand_noise_samples)
                        samples = pd.DataFrame(samples, columns=self.col_names)
                        samples.to_csv(
                            f'data/samples_{epoch}_{self.params["critic_iters"]}_{self.params["base_nodes"]}_synthetic_{i}.csv',
                            index=False)

                # Update log after every 100 epochs
                if epoch < 5 or epoch % 100 == 99:
                    with open(
                            f'log_{self.params["critic_iters"]}_{self.params["base_nodes"]}.pkl',
                            'wb') as f:
                        pkl.dump({
                            'time': self.time_all,
                            'disc_loss': self.disc_loss_all,
                            'gen_loss': self.gen_loss_all,
                            'test_loss': self.disc_loss_test_all
                        }, f)

            saver.save(
                session,
                os.path.join(
                    os.getcwd(),
                    f'model_{self.params["critic_iters"]}_{self.params["base_nodes"]}.ckpt'
                ))

            tf.io.write_graph(session.graph, '.', 'wgan_graph.pbtxt')