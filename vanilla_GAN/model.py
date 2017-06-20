import tensorflow as tf

from initializer import xavier_init


class VanillaGAN:

    def __init__(self, input_dim, z_dim, learning_rate):
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.learning_rate = learning_rate

        self.X = tf.placeholder(tf.float32, shape=[None, input_dim])
        self.Z = tf.placeholder(tf.float32, shape=[None, z_dim])

    def build(self):
        # Training variables for Discriminator
        self.D_W1 = tf.Variable(xavier_init([self.input_dim, 128]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[128]))
        self.D_W2 = tf.Variable(xavier_init([128, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

        # Training variables for Generator
        self.G_W1 = tf.Variable(xavier_init([self.z_dim, 128]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[128]))
        self.G_W2 = tf.Variable(xavier_init([128, self.input_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.input_dim]))

        self.fake_sample = self.generator(self.Z)

        self._loss()
        self._build_ops()

    def generator(self, z):
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)

        return G_prob

    def discriminator(self, x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2

        return D_logit

    def _loss(self):
        D_logit_real = self.discriminator(self.X)
        D_logit_fake = self.discriminator(self.fake_sample)

        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))

        self.d_loss = D_loss_real + D_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

    def _build_ops(self):
        self.d_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss, var_list=[self.D_W1, self.D_b1, self.D_W2, self.D_b2])
        self.g_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss, var_list=[self.G_W1, self.G_b1, self.G_W2, self.G_b2])
