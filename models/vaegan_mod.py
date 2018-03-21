import math
import numpy as np
import tensorflow as tf

from .base import CondBaseModel
from .utils import *

class Encoder(object):
    def __init__(self, input_shape, z_dims, num_attrs):
        self.variables = None
        self.reuse = False
        self.input_shape = input_shape
        self.z_dims = z_dims
        self.num_attrs = num_attrs
        self.name = 'encoder'

    def __call__(self, inputs, attrs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('conv1'):
                a = tf.reshape(attrs, [-1, 1, 1, self.num_attrs])
                a = tf.tile(a, [1, self.input_shape[0], self.input_shape[1], 1])
                x = tf.concat([inputs, a], axis=-1)
                x = tf.layers.conv2d(x, 64, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d(x, 128, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d(x, 256, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv4'):
                x = tf.layers.conv2d(x, 512, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('global_avg'):
                x = tf.reduce_mean(x, axis=[1, 2])

            with tf.variable_scope('fc1'):
                z_avg = tf.layers.dense(x, self.z_dims)
                z_log_var = tf.layers.dense(x, self.z_dims)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.reuse = True

        return z_avg, z_log_var

class Decoder(object):
    def __init__(self, input_shape):
        self.variables = None
        self.reuse = False
        self.input_shape = input_shape
        self.name = 'decoder'

    def __call__(self, inputs, attrs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('fc1'):
                w = self.input_shape[0] // (2 ** 3)
                x = tf.concat([inputs, attrs], axis=-1)
                x = tf.layers.dense(x, w * w * 256)
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)
                x = tf.reshape(x, [-1, w, w, 256])

            with tf.variable_scope('conv1'):
                x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d_transpose(x, 64, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv4'):
                d = self.input_shape[2]
                x = tf.layers.conv2d_transpose(x, d, (5, 5), (1, 1), 'same')
                x = tf.tanh(x)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
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


class VAEGAN_mod(CondBaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        name='vaegan_mod',
        **kwargs
    ):
        super(VAEGAN_mod, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        # Parameters for feature matching
        self.use_feature_match = False
        self.alpha = 0.7

        self.E_f_D_r = None
        self.E_f_D_p = None

        self.f_enc = None
        self.f_gen = None
        self.f_dis = None

        self.x_r = None
        self.z_p = None

        self.z_test = None
        self.x_test = None

        self.enc_trainer = None
        self.gen_trainer = None
        self.dis_trainer = None

        self.gen_loss = None
        self.dis_loss = None
        self.gen_acc = None
        self.dis_acc = None

        self.build_model()

    def train_on_batch(self, batch, index):
        x_r, c_r = batch
        batchsize = len(x_r)
        z_p = np.random.uniform(-1, 1, size=(len(x_r), self.z_dims))

        _, _, _, _, gen_loss, dis_loss, gen_acc, dis_acc = self.sess.run(
            (self.gen_trainer, self.enc_trainer, self.dis_trainer, self.gen_loss,
             self.dis_loss, self.gen_acc, self.dis_acc),
            feed_dict={
                self.x_r: x_r, self.z_p: z_p,
                self.z_test: self.test_data['z_test']})

        summary_priod = 1000
        if index // summary_priod != (index + batchsize) // summary_priod:
            summary = self.sess.run(
                self.summary,
                feed_dict={
                    self.x_r: x_r, self.z_p: z_p, self.c_r: c_r,
                    self.z_test: self.test_data['z_test']})
            self.writer.add_summary(summary, index)

        return [
            ('gen_loss', gen_loss), ('dis_loss', dis_loss),
            ('gen_acc', gen_acc), ('dis_acc', dis_acc)
        ]

    def predict(self, batch):
        z_samples, c_samples = batch
        x_sample = self.sess.run(
            self.x_test,
            feed_dict={self.z_test: z_samples}
        )
        return x_sample

    def make_test_data(self):
        c_t = np.identity(self.num_attrs)
        c_t = np.tile(c_t, (self.test_size, 1))
        z_t = np.random.normal(size=(self.test_size, self.z_dims))
        z_t = np.tile(z_t, (1, self.num_attrs))
        z_t = z_t.reshape((self.test_size * self.num_attrs, self.z_dims))
        self.test_data = {'z_test': z_t}

    def build_model(self):
        self.f_enc = Encoder(self.input_shape, self.z_dims, self.num_attrs)
        self.f_gen = Decoder(self.input_shape)

        n_cls_out = self.num_attrs if self.use_feature_match else self.num_attrs + 1
        self.f_dis = Discriminator(self.input_shape)

        # Trainer
        self.x_r = tf.placeholder(tf.float32, shape=(None,) + self.input_shape)

        z_avg, z_log_var = self.f_enc(self.x_r, self.num_attrs)

        z_f = sample_normal(z_avg, z_log_var)
        x_f = self.f_gen(z_f)

        self.z_p = tf.placeholder(tf.float32, shape=(None, self.z_dims))
        x_p = self.f_gen(self.z_p)

        y_r, f_D_r = self.f_dis(self.x_r)
        y_f, f_D_f = self.f_dis(x_f)
        y_p, f_D_p = self.f_dis(x_p)

        L_KL = kl_loss(z_avg, z_log_var)

        enc_opt = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)
        gen_opt = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)
        dis_opt = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)

        if self.use_feature_match:
            # Use feature matching (it is usually unstable)
            L_GD = self.L_GD(f_D_r, f_D_p)
            L_G = self.L_G(self.x_r, x_f, f_D_r, f_D_f)

            with tf.name_scope('L_D'):
                L_D = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_r), y_r) + \
                      tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_f), y_f) + \
                      tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_p), y_p)

            self.enc_trainer = enc_opt.minimize(L_G + L_KL, var_list=self.f_enc.variables)
            self.gen_trainer = gen_opt.minimize(L_G + L_GD, var_list=self.f_gen.variables)
            self.dis_trainer = dis_opt.minimize(L_D, var_list=self.f_dis.variables)

            self.gen_loss = L_G + L_GD
            self.dis_loss = L_D

            # Predictor
            self.z_test = tf.placeholder(tf.float32, shape=(None, self.z_dims))

            self.x_test = self.f_gen(self.z_test, self.c_test)
            x_tile = self.image_tiling(self.x_test, self.test_size, self.num_attrs)

            # Summary
            tf.summary.image('x_real', self.x_r, 10)
            tf.summary.image('x_fake', x_f, 10)
            tf.summary.image('x_tile', x_tile, 1)
            tf.summary.scalar('L_G', L_G)
            tf.summary.scalar('L_GD', L_GD)
            tf.summary.scalar('L_D', L_D)
            tf.summary.scalar('L_KL', L_KL)
            tf.summary.scalar('gen_loss', self.gen_loss)
            tf.summary.scalar('dis_loss', self.dis_loss)
        else:
            # Not use feature matching (it is more similar to ordinary GANs)
            with tf.name_scope('L_G'):
                L_G = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_f), y_f) + \
                      tf.losses.sigmoid_cross_entropy(tf.ones_like(y_p), y_p)

            with tf.name_scope('L_rec'):
                # L_rec =  0.5 * tf.losses.mean_squared_error(self.x_r, x_f)
                L_rec =  0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.x_r, x_f), axis=[1, 2, 3]))

            with tf.name_scope('L_D'):
                L_D = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_r), y_r) + \
                      tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_f), y_f) + \
                      tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_p), y_p)


            self.enc_trainer = enc_opt.minimize(L_rec + L_KL, var_list=self.f_enc.variables)
            self.gen_trainer = gen_opt.minimize(L_G + L_rec, var_list=self.f_gen.variables)
            self.dis_trainer = dis_opt.minimize(L_D, var_list=self.f_dis.variables)

            self.gen_loss = L_G + L_rec
            self.dis_loss = L_D

            # Predictor
            self.z_test = tf.placeholder(tf.float32, shape=(None, self.z_dims))

            self.x_test = self.f_gen(self.z_test)
            x_tile = self.image_tiling(self.x_test, self.test_size, self.num_attrs)

            # Summary
            tf.summary.image('x_real', self.x_r, 10)
            tf.summary.image('x_fake', x_f, 10)
            tf.summary.image('x_tile', x_tile, 1)
            tf.summary.scalar('L_G', L_G)
            tf.summary.scalar('L_rec', L_rec)
            tf.summary.scalar('L_D', L_D)
            tf.summary.scalar('L_KL', L_KL)
            tf.summary.scalar('gen_loss', self.gen_loss)
            tf.summary.scalar('dis_loss', self.dis_loss)

        # Accuracy
        self.gen_acc = 0.5 * binary_accuracy(tf.ones_like(y_f), y_f) + \
                       0.5 * binary_accuracy(tf.ones_like(y_p), y_p)

        self.dis_acc = binary_accuracy(tf.ones_like(y_r), y_r) / 3.0 + \
                       binary_accuracy(tf.zeros_like(y_f), y_f) / 3.0 + \
                       binary_accuracy(tf.zeros_like(y_p), y_p) / 3.0

        tf.summary.scalar('gen_acc', self.gen_acc)
        tf.summary.scalar('dis_acc', self.dis_acc)

        self.summary = tf.summary.merge_all()

    def L_G(self, x_r, x_f, f_D_r, f_D_f, f_C_r, f_C_f):
        with tf.name_scope('L_G'):
            loss = tf.constant(0.0, dtype=tf.float32)
            loss += 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x_r, x_f), axis=[1, 2, 3]))
            loss += 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(f_D_r, f_D_f), axis=[1]))
            loss += 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(f_C_r, f_C_f), axis=[1]))

        return loss

    def L_GD(self, f_D_r, f_D_p):
        with tf.name_scope('L_GD'):
            # Compute loss
            E_f_D_r = tf.reduce_mean(f_D_r, axis=0)
            E_f_D_p = tf.reduce_mean(f_D_p, axis=0)

            # Update features
            if self.E_f_D_r is None:
                self.E_f_D_r = tf.zeros_like(E_f_D_r)

            if self.E_f_D_p is None:
                self.E_f_D_p = tf.zeros_like(E_f_D_p)

            self.E_f_D_r = self.alpha * self.E_f_D_r + (1.0 - self.alpha) * E_f_D_r
            self.E_f_D_p = self.alpha * self.E_f_D_p + (1.0 - self.alpha) * E_f_D_p
            return 0.5 * tf.reduce_sum(tf.squared_difference(self.E_f_D_r, self.E_f_D_p))