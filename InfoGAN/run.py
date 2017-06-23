import os
import argparse
import sys
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import InfoGAN
from utils import sample_uniform


def plot_loss(iters, d_losses, g_losses):
    plt.figure(figsize=(10, 8))
    plt.plot(d_losses, label='Discriminitive loss')
    plt.plot(g_losses, label='Generative loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('out/loss_iter_%d.png' % iters)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def train(image_size, z_dim, c_dim, hidden_dim, batch_size, num_iterations, learning_rate, print_every=1000):
    # Load data
    mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

    with tf.Graph().as_default():
        sess = tf.Session()

        with sess.as_default():
            # Build model
            model = InfoGAN(image_size, z_dim, c_dim, hidden_dim, learning_rate)
            model.build()

            sess.run(tf.global_variables_initializer())

            if not os.path.exists('out/'):
                os.makedirs('out/')

            d_losses = []
            g_losses = []
            i = 0
            for it in range(num_iterations):
                if it % print_every == 0:
                    z_noise = sample_uniform(16, z_dim)

                    idx = np.random.randint(0, 10)
                    c_noise = np.zeros([16, 10])
                    c_noise[range(16), idx] = 1

                    G_samples = sess.run(model.G_samples,
                                         feed_dict={model.z: z_noise, model.c: c_noise})

                    fig = plot(G_samples)
                    plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    i += 1
                    plt.close(fig)

                X_batch, _ = mnist.train.next_batch(batch_size)
                z_noise = sample_uniform(batch_size, z_dim)
                c_noise = np.random.multinomial(1, 10 * [0.1], size=batch_size)

                _, d_loss = sess.run([model.d_solver, model.d_loss],
                                     feed_dict={model.X: X_batch, model.z: z_noise, model.c: c_noise})
                _, g_loss = sess.run([model.g_solver, model.g_loss],
                                     feed_dict={model.z: z_noise, model.c: c_noise})
                sess.run([model.Q_solver], feed_dict={model.z: z_noise, model.c: c_noise})

                d_losses.append(d_loss)
                g_losses.append(g_loss)

                if it % print_every == 0:
                    print('Iter: {}'.format(it))
                    print('Discriminator Loss: {:.4}\tGenerator Loss: {:.4}\n'.format(d_loss, g_loss))

            plot_loss(it, d_losses, g_losses)


def run(_):
    if FLAGS.mode == 'train':
        image_size = 28 * 28 * 1
        train(image_size, FLAGS.z_dim, FLAGS.c_dim, FLAGS.hidden_dim, FLAGS.batch_size, FLAGS.num_iterations, FLAGS.learning_rate)
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--z_dim',
        type=int,
        default=16,
        help='Specify the dimension of z.'
    )
    parser.add_argument(
        '--c_dim',
        type=int,
        default=10,
        help='Specify the dimension of c.'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=128,
        help='Specify the dimension of hidden layer.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-4,
        help='Specify learning rate'
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=10000,
        help='Specify number of iterations'
    )
    parser.add_argument(
        '--mode',
        type=str,
        help='Specify mode: `train` or `eval`',
        required=True
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)
