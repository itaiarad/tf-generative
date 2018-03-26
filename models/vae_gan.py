import tensorflow as tf
import numpy as np

from .base import BaseModel, HandBaseModel
from .utils import *
from .wnorm import *


class Encoder(object):
    def __init__(self, input_shape, z_dims, use_wnorm=True):
        self.variables = None
        self.update_ops = None
        self.reuse = False
        self.input_shape = input_shape
        self.z_dims = z_dims
        self.use_wnorm = use_wnorm

    def __call__(self, inputs, training=True):
        with tf.variable_scope('encoder', reuse=self.reuse):
            with tf.variable_scope('conv1'):
                if self.use_wnorm:
                    x = conv2d_wnorm(inputs, 64, (5, 5), (2, 2), 'same', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d(inputs, 64, (5, 5), (2, 2), 'same')
                    x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv2'):
                if self.use_wnorm:
                    x = conv2d_wnorm(x, 128, (5, 5), (2, 2), 'same', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d(x, 128, (5, 5), (2, 2), 'same')
                    x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv3'):
                if self.use_wnorm:
                    x = conv2d_wnorm(x, 256, (5, 5), (2, 2), 'same', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d(x, 256, (5, 5), (2, 2), 'same')
                    x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv4'):
                if self.use_wnorm:
                    x = conv2d_wnorm(x, 512, (5, 5), (2, 2), 'same', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d(x, 512, (5, 5), (2, 2), 'same')
                    x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('fc1'):
                w = self.input_shape[0] // (2 ** 4)
                if self.use_wnorm:
                    z_avg = conv2d_wnorm(x, self.z_dims, (w, w), (1, 1), 'valid', use_scale=True)
                    z_log_var = conv2d_wnorm(x, self.z_dims, (w, w), (1, 1), 'valid', use_scale=True)
                else:
                    z_avg = tf.layers.conv2d(x, self.z_dims, (w, w), (1, 1), 'valid')
                    z_log_var = tf.layers.conv2d(x, self.z_dims, (w, w), (1, 1), 'valid')

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='encoder')
        self.reuse = True

        return z_avg, z_log_var


class Decoder(object):
    def __init__(self, input_shape, z_dims, use_wnorm=True):
        self.variables = None
        self.update_ops = None
        self.reuse = False
        self.input_shape = input_shape
        self.z_dims = z_dims
        self.use_wnorm = use_wnorm

    def __call__(self, inputs, training=True):
        with tf.variable_scope('decoder', reuse=self.reuse):
            with tf.variable_scope('deconv1'):
                w = self.input_shape[0] // (2 ** 3)
                x = tf.reshape(inputs, [-1, 1, 1, self.z_dims])
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, 256, (w, w), (1, 1), 'valid', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d_transpose(x, 256, (w, w), (1, 1), 'valid')
                    x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv2'):
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, 256, (5, 5), (2, 2), 'same', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2), 'same')
                    x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv3'):
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, 128, (5, 5), (2, 2), 'same', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2), 'same')
                    x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv4'):
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, 64, (5, 5), (2, 2), 'same', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d_transpose(x, 64, (5, 5), (2, 2), 'same')
                    x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv5'):
                d = self.input_shape[2]
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, d, (5, 5), (1, 1), 'same', use_scale=True)
                else:
                    x = tf.layers.conv2d_transpose(x, d, (5, 5), (1, 1), 'same')
                x = tf.tanh(x)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='decoder')
        self.reuse = True

        return x


class Discriminator(object):
    def __init__(self, input_shape):
        self.variables = None
        self.reuse = False
        self.input_shape = input_shape
        self.name = 'discriminator'

    def __call__(self, inputs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('conv1'):
                x = tf.layers.conv2d(inputs, 64, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d(x, 128, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d(x, 256, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv4'):
                x = tf.layers.conv2d(x, 512, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('global_avg'):
                x = tf.reduce_mean(x, axis=[1, 2])

            with tf.variable_scope('fc1'):
                f = tf.contrib.layers.flatten(x)
                y = tf.layers.dense(f, 1)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.reuse = True

        return y, f


class VAEGAN(HandBaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        name='vaegan',
        **kwargs
    ):
        super(VAEGAN, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims
        self.use_wnorm = False
        self.use_noise = False
        self.encoder = None
        self.decoder = None
        self.rec_loss = None
        self.kl_loss = None
        self.train_op = None

        self.x_train_real = None

        self.z_test = None
        self.x_test = None

        #  Added discriminator parameters
        self.use_feature_match = kwargs['use_feature_match']
        self.E_f_D_real = None
        self.E_f_D_fake = None
        self.alpha = kwargs['alpha_decay']
        self.gamma_img2img = kwargs['gamma_img2img']  #  L_G loss ratio from img to img
        self.gamma_dis = kwargs['gamma_dis']  #  L_G loss ratio from discriminator

        #  General training parameters
        self.lr_enc = kwargs['lr_enc']
        self.lr_dec = kwargs['lr_dec']
        self.lr_dis = kwargs['lr_dis']
        self.beta1_enc = kwargs['beta1_enc']
        self.beta1_dec = kwargs['beta1_dec']
        self.beta1_dis = kwargs['beta1_dis']

        self.build_model()

    def train_on_batch(self, x_batch, index, z_p=None):
        batchsize = len(x_batch)
        if z_p is None:
            z_p = np.random.uniform(-1, 1, size=(len(x_batch), self.z_dims))

        _, _, _, kl_loss, gen_loss, dis_loss, gen_acc, dis_acc, summary = self.sess.run(
            (self.gen_trainer, self.enc_trainer, self.dis_trainer, self.kl_loss, self.gen_loss,
             self.dis_loss, self.gen_acc, self.dis_acc, self.summary),
            feed_dict={
                self.x_train_real: x_batch, self.z_p: z_p,
                self.z_test: self.test_data})

        summary_priod = 1000
        if index // summary_priod != (index + batchsize) // summary_priod:
            summary = self.sess.run(
                self.summary,
                feed_dict={
                    self.x_train_real: x_batch, self.z_p: z_p, self.z_test: self.test_data})
            self.writer.add_summary(summary, index)

        return [
            ('gen_loss', gen_loss), ('dis_loss', dis_loss),
            ('gen_acc', gen_acc), ('dis_acc', dis_acc), ('kl_loss', kl_loss)]

    def predict(self, z_samples):
        x_sample = self.sess.run(
            self.x_test,
            feed_dict={self.z_test: z_samples}
        )
        return x_sample

    def make_test_data(self):
        self.test_data = np.random.normal(size=(self.test_size * self.test_size, self.z_dims))

    def build_model(self):
        self.encoder = Encoder(self.input_shape, self.z_dims, self.use_wnorm)
        self.decoder = Decoder(self.input_shape, self.z_dims, self.use_wnorm)
        self.discriminator = Discriminator(self.input_shape)

        # Trainer
        batch_shape = (None,) + self.input_shape
        self.x_train_real = tf.placeholder(tf.float32, shape=batch_shape)

        z_avg, z_log_var = self.encoder(self.x_train_real)
        z_sample = sample_normal(z_avg, z_log_var)
        x_sample = self.decoder(z_sample)

        if self.use_noise:
            self.z_p = tf.placeholder(tf.float32, shape=(None, self.z_dims))
        else:
            self.z_p = tf.placeholder(tf.float32, shape=batch_shape)

        x_p = self.decoder(self.z_p)

        y_real, f_D_real = self.discriminator(self.x_train_real)
        y_fake, f_D_fake = self.discriminator(x_sample)
        y_p, f_D_p = self.discriminator(x_p)

        rec_loss_scale = tf.constant(np.prod(self.input_shape), tf.float32)
        self.rec_loss = tf.losses.absolute_difference(self.x_train_real, x_sample) * rec_loss_scale
        self.kl_loss = kl_loss(z_avg, z_log_var)

        enc_optim = tf.train.AdamOptimizer(learning_rate=self.lr_enc, beta1=self.lr_enc)
        dec_optim = tf.train.AdamOptimizer(learning_rate=self.lr_dec, beta1=self.lr_dec)
        dis_optim = tf.train.AdamOptimizer(learning_rate=self.lr_dis, beta1=self.lr_dis)
        optim = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)

        if self.use_feature_match:
            L_GD = self.L_GD(f_D_real, f_D_p)
            L_G = self.L_G(self.x_train_real, x_sample, f_D_real, f_D_fake)
            with tf.name_scope('L_D'):
                L_D = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_real), y_real) + \
                      tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake), y_fake) + \
                      tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_p), y_p)

            self.enc_trainer = enc_optim.minimize(L_G + self.kl_loss, var_list=self.encoder.variables)
            self.gen_trainer = dec_optim.minimize(L_G + L_GD, var_list=self.decoder.variables)
            self.dis_trainer = dis_optim.minimize(L_D, var_list=self.discriminator.variables)

            self.gen_loss = L_G + L_GD
            self.dis_loss = L_D

            tf.summary.scalar('L_GD', L_GD)
        else:
            with tf.name_scope('L_G'):
                L_G = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_fake), y_fake) + \
                      tf.losses.sigmoid_cross_entropy(tf.ones_like(y_p), y_p)

            with tf.name_scope('L_rec'):
                L_rec =  0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.x_train_real, x_sample), axis=[1, 2, 3]))

            with tf.name_scope('L_D'):
                L_D = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_real), y_real) + \
                      tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake), y_fake) + \
                      tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_p), y_p)

            self.enc_trainer = enc_optim.minimize(L_rec + self.kl_loss, var_list=self.encoder.variables)
            self.gen_trainer = dec_optim.minimize(L_G + L_rec, var_list=self.decoder.variables)
            self.dis_trainer = dis_optim.minimize(L_D, var_list=self.discriminator.variables)

            self.gen_loss = L_G + L_rec
            self.dis_loss = L_D

        # Accuracy
        self.gen_acc = 0.5 * binary_accuracy(tf.ones_like(y_fake), y_fake) + \
                       0.5 * binary_accuracy(tf.ones_like(y_p), y_p)

        self.dis_acc = binary_accuracy(tf.ones_like(y_real), y_real) / 3.0 + \
                       binary_accuracy(tf.zeros_like(y_fake), y_fake) / 3.0 + \
                       binary_accuracy(tf.zeros_like(y_p), y_p) / 3.0


        # fmin = optim.minimize(self.rec_loss + self.kl_loss)  #  can erase later
        #
        # with tf.control_dependencies([fmin] + self.encoder.update_ops + self.decoder.update_ops):
        #     self.train_op = tf.no_op(name='train')

        # Predictor
        self.z_test = tf.placeholder(tf.float32, shape=(None, self.z_dims))
        self.x_test = self.decoder(self.z_test)
        x_tile = self.image_tiling(self.x_test, self.test_size, self.test_size)

        # Summary
        tf.summary.image('x_real', image_cast(self.x_train_real), 10)
        tf.summary.image('x_fake', image_cast(x_sample), 10)
        tf.summary.image('x_tile', image_cast(x_tile), 1)
        tf.summary.scalar('rec_loss', self.rec_loss)
        tf.summary.scalar('kl_loss', self.kl_loss)
        # Added summary
        tf.summary.scalar('gen_loss', self.gen_loss)
        tf.summary.scalar('L_G', L_G)
        tf.summary.scalar('L_D', L_D)
        tf.summary.scalar('L_KL', self.kl_loss)
        tf.summary.scalar('dis_loss', self.dis_loss)
        tf.summary.scalar('gen_acc', self.gen_acc)
        tf.summary.scalar('dis_acc', self.dis_acc)
        # tf.summary.histogram('encoder_variables', self.encoder.variables)
        # tf.summary.histogram('decoder_variables', self.decoder.variables)
        # tf.summary.histogram('discriminator_variables', self.discriminator.variables)

        self.summary = tf.summary.merge_all()

    def L_G(self, x_r, x_f, f_D_r, f_D_f):
        with tf.name_scope('L_G'):
            loss = tf.constant(0.0, dtype=tf.float32)
            loss += self.gamma_img2img * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x_r, x_f), axis=[1, 2, 3]))
            loss += self.gamma_dis * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(f_D_r, f_D_f), axis=[1]))

        return loss

    def L_GD(self, f_D_r, f_D_p):
        with tf.name_scope('L_GD'):
            # Compute loss
            E_f_D_r = tf.reduce_mean(f_D_r, axis=0)
            E_f_D_p = tf.reduce_mean(f_D_p, axis=0)

            # Update features
            if self.E_f_D_real is None:
                self.E_f_D_real = tf.zeros_like(E_f_D_r)

            if self.E_f_D_fake is None:
                self.E_f_D_fake = tf.zeros_like(E_f_D_p)

            self.E_f_D_real = self.alpha * self.E_f_D_real + (1.0 - self.alpha) * E_f_D_r
            self.E_f_D_fake = self.alpha * self.E_f_D_fake + (1.0 - self.alpha) * E_f_D_p
            return 0.5 * tf.reduce_sum(tf.squared_difference(self.E_f_D_real, self.E_f_D_fake))