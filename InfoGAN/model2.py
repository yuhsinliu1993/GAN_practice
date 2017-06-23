import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope


def leaky_relu(x):
    return tf.where(tf.greater(x, 0), x, 0.01 * x)


class InfoGAN:

    def __init__(self, x_dim, z_dim=38, num_category=10, num_continue=2, learning_rate=1e-4):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.num_category = num_category
        self.num_continue = num_continue
        self.learning_rate = learning_rate

        self.X = tf.placeholder(tf.float32, shape=[None, x_dim])
        self.z_category = tf.placeholder(tf.float32, shape=[None, num_category])
        self.z_continue = tf.placeholder(tf.float32, shape=[None, num_continue])
        self.z_noise = tf.placeholder(tf.float32, shape=[None, z_dim])

    def build(self):
        self.z_c_noise = tf.concat(values=[self.z_category, self.z_continue, self.z_noise], axis=1)
        self.G_samples = self.generator(self.z_c_noise)

        self.D_real_logits, _, _ = self.discriminator(self.X)
        self.D_fake_logits, self.D_fake_category, self.D_fake_continue = self.discriminator(self.G_samples)

        self._loss()
        self._build_ops()

    def generator(self, z_c):
        reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0

        with variable_scope.variable_scope('generator', reuse=reuse):
            G_hidden = slim.fully_connected(z_c, 1024)
            G_hidden = slim.batch_norm(G_hidden, activation_fn=tf.nn.relu)
            G_hidden = slim.fully_connected(G_hidden, 7 * 7 * 128)
            G_hidden = slim.batch_norm(G_hidden, activation_fn=tf.nn.relu)
            G_hidden = tf.reshape(G_hidden, [-1, 7, 7, 128])
            G_hidden = slim.conv2d_transpose(G_hidden, 64, kernel_size=[4, 4], stride=2, activation_fn=None)
            G_hidden = slim.batch_norm(G_hidden, activation_fn=tf.nn.relu)
            G_samples = slim.conv2d_transpose(G_hidden, 1, kernel_size=[4, 4], stride=2, activation_fn=tf.nn.sigmoid)

        return G_samples

    def discriminator(self, x):
        reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0

        with variable_scope.variable_scope('discriminator', reuse=reuse):
            x = tf.reshape(x, [-1, 28, 28, 1])
            x = slim.conv2d(x, num_outputs=64, kernel_size=[4, 4], stride=2, activation_fn=leaky_relu)
            x = slim.conv2d(x, num_outputs=128, kernel_size=[4, 4], stride=2, activation_fn=leaky_relu)
            x = slim.flatten(x)

            shared_x = slim.fully_connected(x, num_outputs=1024, activation_fn=leaky_relu)
            d_logits = slim.fully_connected(shared_x, num_outputs=1, activation_fn=None)
            d_logits = tf.squeeze(d_logits, -1)

            recog_shared = slim.fully_connected(shared_x, num_outputs=128, activation_fn=leaky_relu)
            recog_category = slim.fully_connected(recog_shared, num_outputs=self.num_category, activation_fn=None)
            recog_continue = slim.fully_connected(recog_shared, num_outputs=self.num_continue, activation_fn=tf.nn.sigmoid)

        return d_logits, recog_category, recog_continue

    def _loss(self):
        self.d_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(self.D_real_logits) + 1e-8) + tf.log(1 - tf.nn.sigmoid(self.D_fake_logits) + 1e-8))
        self.g_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(self.D_fake_logits) + 1e-8))

        self.category_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_fake_category, labels=self.z_category))

        self.continue_loss = tf.reduce_mean(tf.square(self.D_fake_continue - self.z_continue))

    def _build_ops(self):
        theta_D = [t for t in tf.trainable_variables() if t.name.startswith('discriminator')]
        theta_G = [t for t in tf.trainable_variables() if t.name.startswith('generator')]

        self.D_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss + self.category_loss + self.continue_loss, var_list=theta_D)
        self.G_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss + self.category_loss + self.continue_loss, var_list=theta_G)
