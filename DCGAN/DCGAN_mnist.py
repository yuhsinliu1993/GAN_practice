import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializations


# In 'th' mode, the channels dimension (the depth) is at index 1
K.set_image_dim_ordering('th')

np.random.seed(1000)

random_dim = 100

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train[:, np.newaxis, :, :]


adam = Adam(lr=0.0002, beta_1=0.5)

# Generator
generator = Sequential()
generator.add(Dense(128 * 7 * 7, input_dim=random_dim, kernel_initializer=random_uniform))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((128, 7, 7)))
generator.add(UpSampling2D(size=(2, 2), data_format='channels_first'))
generator.add(Conv2D(64, 5, 5, border_mode='same'))
generator.add(LeakyReLU(0.2))
generator.add(UpSampling2D(size=(2, 2), data_format='channels_first'))
generator.add(Convolution2D(1, 5, 5, border_mode='same', activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)


# Discriminator
discriminator = Sequential()
discriminator.add(Convolution2D(64, 5, 5, border_mode='same', subsample=(2, 2), input_shape=(1, 28, 28), kernel_initializer=random_uniform))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Convolution2D(128, 5, 5, border_mode='same', subsample=(2, 2)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)


# Combined network
discriminator.trainable = False
gan_input = Input(shape=(random_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(inputs=gan_input, outputs=gan_output)
gan.compile(loss='binary_crossentropy', optimizer=adam)


d_losses = []
g_losses = []


# Plot the loss from each batch
def plot_loss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(d_losses, label='Discriminitive loss')
    plt.plot(g_losses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/dcgan_loss_epoch_%d.png' % epoch)


# Create a wall of generated MNIST images
def plot_generated_images(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/dcgan_generated_image_epoch_%d.png' % epoch)


# Save the generator and discriminator networks (and weights) for later use
def save_models(epoch):
    generator.save('models/dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/dcgan_discriminator_epoch_%d.h5' % epoch)


def Least_Square_loss():
    pass


def train(epochs=1, batch_size=128):
    total_batch_number = X_train.shape[0] / batch_size

    print 'Number of epochs: ', epochs
    print 'Batch size: ', batch_size
    print 'Batches per epoch: ', total_batch_number

    for e in xrange(1, epochs + 1):
        print '-' * 15, 'Epoch %d' % e, '-' * 15

        for _ in tqdm(xrange(total_batch_number)):

            # Get a random set of input images
            batch_indices = np.random.randint(0, X_train.shape[0], size=batch_size)
            real_images = X_train[batch_indices]

            # Generate fake MNIST images
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            generated_images = generator.predict(noise)

            X = np.concatenate([real_images, generated_images])

            # Labels for generated and real data
            yDis = np.zeros(2 * batch_size)
            # One-sided label smoothing
            yDis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            yGen = np.ones(batch_size)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, yGen)

        # Store loss of most recent batch from this epoch
        d_losses.append(d_loss)
        g_losses.append(g_loss)

        if e == 1 or e % 5 == 0:
            plot_generated_images(e)
            save_models(e)

    # Plot losses from every epoch
    plot_loss(e)


if __name__ == '__main__':
    train(10, 128)
