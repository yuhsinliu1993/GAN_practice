import tensorflow as tf

from initializer import xavier_init


class InfoGAN:

    def __init__(self, x_dim, z_dim, c_dim, hidden_dim, learning_rate):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        self.X = tf.placeholder(tf.float32, shape=[None, x_dim])
        self.z = tf.placeholder(tf.float32, shape=[None, z_dim])
        self.c = tf.placeholder(tf.float32, shape=[None, c_dim])

    def build(self):
        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_samples)

        self.G_samples = self.generator(self.z, self.c)
        self.Q_c_given_x = self.Q(self.G_samples)

        self._loss()
        self._build_ops()

    def generator(self, z, c):
        self.G_W1 = tf.Variable(xavier_init([self.z_dim + self.c_dim, 2 * self.hidden_dim]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[2 * self.hidden_dim]))
        self.G_W2 = tf.Variable(xavier_init([2 * self.hidden_dim, self.x_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.x_dim]))
        self.theta_G = [self.G_W1, self.G_b1, self.G_W2, self.G_b2]

        z_c = tf.concat(values=[z, c], axis=1)

        G_h1 = tf.nn.relu(tf.matmul(z_c, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)

        return G_prob

    def discriminator(self, x):
        self.D_W1 = tf.Variable(xavier_init([self.x_dim, self.hidden_dim]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.hidden_dim]))
        self.D_W2 = tf.Variable(xavier_init([self.hidden_dim, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))
        self.theta_D = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob

    def Q(self, x):
        self.Q_W1 = tf.Variable(xavier_init([self.x_dim, self.hidden_dim]))
        self.Q_b1 = tf.Variable(tf.zeros(shape=[self.hidden_dim]))
        self.Q_W2 = tf.Variable(xavier_init([self.hidden_dim, 10]))
        self.Q_b2 = tf.Variable(tf.zeros(shape=[10]))
        self.theta_Q = [self.Q_W1, self.Q_b1, self.Q_W2, self.Q_b2]

        Q_h1 = tf.nn.relu(tf.matmul(x, self.Q_W1) + self.Q_b1)
        Q_prob = tf.nn.softmax(tf.matmul(Q_h1, self.Q_W2) + self.Q_b2)

        return Q_prob

    def _loss(self):
        self.d_loss = -tf.reduce_mean(tf.log(self.D_real + 1e-8) + tf.log(1 - self.D_fake + 1e-8))
        self.g_loss = -tf.reduce_mean(tf.log(self.D_fake + 1e-8))

        # LI (G, Q) = Ec~P (c),x~G(z,c)[log Q(c|x)] + H(c)
        cross_ent = tf.reduce_mean(-tf.reduce_sum(tf.log(self.Q_c_given_x + 1e-8) * self.c, 1))
        ent = tf.reduce_mean(-tf.reduce_sum(tf.log(self.c + 1e-8) * self.c, 1))
        self.Q_loss = cross_ent + ent

    def _build_ops(self):

        self.d_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss, var_list=self.theta_D)
        self.g_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss, var_list=self.theta_G)
        self.Q_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.Q_loss, var_list=self.theta_G + self.theta_Q)
