#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import time
import tensorflow as tf

def weight_variable(shape):
    """Xavier initialization"""
    stddev = np.sqrt(2.0 / (sum(shape)))
    initial = tf.truncated_normal(shape, stddev=stddev)
    weights = tf.Variable(initial)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
    return weights

# Bias weights for layers not followed by BatchNorm
# We do not regularlize biases, so they are not
# added to the regularlizer collection
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW',
                        strides=[1, 1, 1, 1], padding='SAME')


# Restore session from checkpoint. It silently ignore mis-matches
# between the checkpoint and the graph. Specifically
# 1. values in the checkpoint for which there is no corresponding variable.
# 2. variables in the graph for which there is no specified value in the
#    checkpoint.
# 3. values where the checkpoint shape differs from the variable shape.
# (variables without a value in the checkpoint are left at their default
# initialized value)
def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    for var_name, saved_var_name in var_names:
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)

class TFProcess:
    def __init__(self):
        # Network structure
        self.RESIDUAL_FILTERS = 128
        self.RESIDUAL_BLOCKS = 6

        # For exporting
        self.weights = []

        # Output weight file with averaged weights
        self.swa_enabled = True
        # Net sampling rate (e.g 2 == every 2nd network).
        self.swa_c = 1
        # Take an exponentially weighted moving average over this
        # many networks. Under the SWA assumptions, this will reduce
        # the distance to the optimal value by a factor of 1/sqrt(n)
        self.swa_max_n = 16
        # Recalculate SWA weight batchnorm means and variances
        self.swa_recalc_bn = True

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.session = tf.Session(config=config)

        self.training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def init(self, dataset, train_iterator, test_iterator):
        # TF variables
        self.handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            self.handle, dataset.output_types, dataset.output_shapes)
        self.next_batch = iterator.get_next()
        self.train_handle = self.session.run(train_iterator.string_handle())
        self.test_handle = self.session.run(test_iterator.string_handle())
        self.init_net(self.next_batch)

    def init_net(self, next_batch):
        self.x = next_batch[0]  # tf.placeholder(tf.float32, [None, 18, 19 * 19])
        self.y_ = next_batch[1] # tf.placeholder(tf.float32, [None, 362])
        self.z_ = next_batch[2] # tf.placeholder(tf.float32, [None, 1])
        self.batch_norm_count = 0
        self.y_conv, self.z_conv = self.construct_net(self.x)

        if self.swa_enabled == True:
            # Count of networks accumulated into SWA
            self.swa_count = tf.Variable(0., name='swa_count', trainable=False)
            # Count of networks to skip
            self.swa_skip = tf.Variable(self.swa_c, name='swa_skip', trainable=False)
            # Build the SWA variables and accumulators
            accum=[]
            load=[]
            n = self.swa_count
            for w in self.weights:
                name = w.name.split(':')[0]
                var = tf.Variable(tf.zeros(shape=w.shape), name='swa/'+name, trainable=False)
                accum.append(tf.assign(var, var * (n / (n + 1.)) + w * (1. / (n + 1.))))
                load.append(tf.assign(w, var))
            with tf.control_dependencies(accum):
                self.swa_accum_op = tf.assign_add(n, 1.)
            self.swa_load_op = tf.group(*load)

        # Calculate loss on policy head
        cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                    logits=self.y_conv)
        self.policy_loss = tf.reduce_mean(cross_entropy)

        # Loss on value head
        self.mse_loss = \
            tf.reduce_mean(tf.squared_difference(self.z_, self.z_conv))

        # Regularizer
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_term = \
            tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        # For training from a (smaller) dataset of strong players, you will
        # want to reduce the factor in front of self.mse_loss here.
        self.loss = 1.0 * self.policy_loss + 1.0 * self.mse_loss + self.reg_term

        # You need to change the learning rate here if you are training
        # from a self-play training set, for example start with 0.005 instead.
        opt_op = tf.train.MomentumOptimizer(
            learning_rate=0.05, momentum=0.9, use_nesterov=True)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = \
                opt_op.minimize(self.loss, global_step=self.global_step)

        correct_prediction = \
            tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        self.avg_policy_loss = []
        self.avg_mse_loss = []
        self.avg_reg_term = []
        self.time_start = None

        # Summary part
        self.test_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "leelalogs/test"), self.session.graph)
        self.train_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "leelalogs/train"), self.session.graph)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        self.session.run(self.init)

    def replace_weights(self, new_weights):
        for e, weights in enumerate(self.weights):
            if weights.name.endswith('/batch_normalization/beta:0'):
                # Batch norm beta is written as bias before the batch normalization
                # in the weight file for backwards compatibility reasons.
                bias = tf.constant(new_weights[e], shape=weights.shape)
                # Weight file order: bias, means, variances
                var = tf.constant(new_weights[e + 2], shape=weights.shape)
                new_beta = tf.divide(bias, tf.sqrt(var + tf.constant(1e-5)))
                self.session.run(tf.assign(weights, new_beta))
            elif weights.shape.ndims == 4:
                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weights.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.session.run(weights.assign(tf.transpose(new_weight, [2, 3, 1, 0])))
            elif weights.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weights.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.session.run(weights.assign(tf.transpose(new_weight, [1, 0])))
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weights[e], shape=weights.shape)
                self.session.run(tf.assign(weights, new_weight))
        #This should result in identical file to the starting one
        #self.save_leelaz_weights('restored.txt')

    def restore(self, file):
        print("Restoring from {0}".format(file))
        optimistic_restore(self.session, file)

    def process(self, batch_size):
        if not self.time_start:
            self.time_start = time.time()
        # Run training for this batch
        policy_loss, mse_loss, reg_term, _, _ = self.session.run(
            [self.policy_loss, self.mse_loss, self.reg_term, self.train_op,
                self.next_batch],
            feed_dict={self.training: True, self.handle: self.train_handle})
        steps = tf.train.global_step(self.session, self.global_step)
        # Keep running averages
        # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
        # get comparable values.
        mse_loss = mse_loss / 4.0
        self.avg_policy_loss.append(policy_loss)
        self.avg_mse_loss.append(mse_loss)
        self.avg_reg_term.append(reg_term)
        if steps % 1000 == 0:
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                speed = batch_size * (1000.0 / elapsed)
            avg_policy_loss = np.mean(self.avg_policy_loss or [0])
            avg_mse_loss = np.mean(self.avg_mse_loss or [0])
            avg_reg_term = np.mean(self.avg_reg_term or [0])
            print("step {}, policy={:g} mse={:g} reg={:g} total={:g} ({:g} pos/s)".format(
                steps, avg_policy_loss, avg_mse_loss, avg_reg_term,
                # Scale mse_loss back to the original to reflect the actual
                # value being optimized.
                # If you changed the factor in the loss formula above, you need
                # to change it here as well for correct outputs.
                avg_policy_loss + 1.0 * 4.0 * avg_mse_loss + avg_reg_term,
                speed))
            train_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Policy Loss", simple_value=avg_policy_loss),
                tf.Summary.Value(tag="MSE Loss", simple_value=avg_mse_loss)])
            self.train_writer.add_summary(train_summaries, steps)
            self.time_start = time_end
            self.avg_policy_loss, self.avg_mse_loss, self.avg_reg_term = [], [], []
        if steps % 8000 == 0:
            sum_accuracy = 0
            sum_mse = 0
            sum_policy = 0
            test_batches = 800
            for _ in range(0, test_batches):
                test_policy, test_accuracy, test_mse, _ = self.session.run(
                    [self.policy_loss, self.accuracy, self.mse_loss,
                     self.next_batch],
                    feed_dict={self.training: False,
                               self.handle: self.test_handle})
                sum_accuracy += test_accuracy
                sum_mse += test_mse
                sum_policy += test_policy
            sum_accuracy /= test_batches
            sum_policy /= test_batches
            # Additionally rescale to [0, 1] so divide by 4
            sum_mse /= (4.0 * test_batches)
            test_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy", simple_value=sum_accuracy),
                tf.Summary.Value(tag="Policy Loss", simple_value=sum_policy),
                tf.Summary.Value(tag="MSE Loss", simple_value=sum_mse)])
            self.test_writer.add_summary(test_summaries, steps)
            print("step {}, policy={:g} training accuracy={:g}%, mse={:g}".\
                format(steps, sum_policy, sum_accuracy*100.0, sum_mse))

            path = os.path.join(os.getcwd(), "leelaz-model")
            leela_path = path + "-" + str(steps) + ".txt"
            self.save_leelaz_weights(leela_path)
            print("Leela weights saved to {}".format(leela_path))

            if self.swa_enabled:
                self.save_swa_network(steps, path, leela_path)

            save_path = self.saver.save(self.session, path, global_step=steps)
            print("Model saved in file: {}".format(save_path))


    def save_leelaz_weights(self, filename):
        with open(filename, "w") as file:
            # Version tag
            file.write("1")
            for weights in self.weights:
                # Newline unless last line (single bias)
                file.write("\n")
                work_weights = None
                if weights.name.endswith('/batch_normalization/beta:0'):
                    # Batch norm beta needs to be converted to biases before
                    # the batch norm for backwards compatibility reasons
                    var_key = weights.name.replace('beta', 'moving_variance')
                    var = tf.get_default_graph().get_tensor_by_name(var_key)
                    work_weights = tf.multiply(weights, tf.sqrt(var + tf.constant(1e-5)))
                elif weights.shape.ndims == 4:
                    # Convolution weights need a transpose
                    #
                    # TF (kYXInputOutput)
                    # [filter_height, filter_width, in_channels, out_channels]
                    #
                    # Leela/cuDNN/Caffe (kOutputInputYX)
                    # [output, input, filter_size, filter_size]
                    work_weights = tf.transpose(weights, [3, 2, 0, 1])
                elif weights.shape.ndims == 2:
                    # Fully connected layers are [in, out] in TF
                    #
                    # [out, in] in Leela
                    #
                    work_weights = tf.transpose(weights, [1, 0])
                else:
                    # Biases, batchnorm etc
                    work_weights = weights
                nparray = work_weights.eval(session=self.session)
                wt_str = [str(wt) for wt in np.ravel(nparray)]
                file.write(" ".join(wt_str))

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def conv_block(self, inputs, filter_size, input_channels, output_channels):
        W_conv = weight_variable([filter_size, filter_size,
                                  input_channels, output_channels])
        # The weights are internal to the batchnorm layer, so apply
        # a unique scope that we can store, and use to look them back up
        # later on.
        weight_key = self.get_batchnorm_key()

        with tf.variable_scope(weight_key):
            h_bn = \
                tf.layers.batch_normalization(
                    conv2d(inputs, W_conv),
                    epsilon=1e-5, axis=1, fused=True,
                    center=True, scale=False,
                    training=self.training)
        h_conv = tf.nn.relu(h_bn)

        beta_key = weight_key + "/batch_normalization/beta:0"
        mean_key = weight_key + "/batch_normalization/moving_mean:0"
        var_key = weight_key + "/batch_normalization/moving_variance:0"

        beta = tf.get_default_graph().get_tensor_by_name(beta_key)
        mean = tf.get_default_graph().get_tensor_by_name(mean_key)
        var = tf.get_default_graph().get_tensor_by_name(var_key)

        self.weights.append(W_conv)
        self.weights.append(beta)
        self.weights.append(mean)
        self.weights.append(var)

        return h_conv

    def residual_block(self, inputs, channels):
        # First convnet
        orig = tf.identity(inputs)
        W_conv_1 = weight_variable([3, 3, channels, channels])
        weight_key_1 = self.get_batchnorm_key()

        # Second convnet
        W_conv_2 = weight_variable([3, 3, channels, channels])
        weight_key_2 = self.get_batchnorm_key()

        with tf.variable_scope(weight_key_1):
            h_bn1 = \
                tf.layers.batch_normalization(
                    conv2d(inputs, W_conv_1),
                    epsilon=1e-5, axis=1, fused=True,
                    center=True, scale=False,
                    training=self.training)
        h_out_1 = tf.nn.relu(h_bn1)
        with tf.variable_scope(weight_key_2):
            h_bn2 = \
                tf.layers.batch_normalization(
                    conv2d(h_out_1, W_conv_2),
                    epsilon=1e-5, axis=1, fused=True,
                    center=True, scale=False,
                    training=self.training)
        h_out_2 = tf.nn.relu(tf.add(h_bn2, orig))

        beta_key_1 = weight_key_1 + "/batch_normalization/beta:0"
        mean_key_1 = weight_key_1 + "/batch_normalization/moving_mean:0"
        var_key_1 = weight_key_1 + "/batch_normalization/moving_variance:0"

        beta_1 = tf.get_default_graph().get_tensor_by_name(beta_key_1)
        mean_1 = tf.get_default_graph().get_tensor_by_name(mean_key_1)
        var_1 = tf.get_default_graph().get_tensor_by_name(var_key_1)

        beta_key_2 = weight_key_2 + "/batch_normalization/beta:0"
        mean_key_2 = weight_key_2 + "/batch_normalization/moving_mean:0"
        var_key_2 = weight_key_2 + "/batch_normalization/moving_variance:0"

        beta_2 = tf.get_default_graph().get_tensor_by_name(beta_key_2)
        mean_2 = tf.get_default_graph().get_tensor_by_name(mean_key_2)
        var_2 = tf.get_default_graph().get_tensor_by_name(var_key_2)

        self.weights.append(W_conv_1)
        self.weights.append(beta_1)
        self.weights.append(mean_1)
        self.weights.append(var_1)

        self.weights.append(W_conv_2)
        self.weights.append(beta_2)
        self.weights.append(mean_2)
        self.weights.append(var_2)

        return h_out_2

    def construct_net(self, planes):
        # NCHW format
        # batch, 18 channels, 19 x 19
        x_planes = tf.reshape(planes, [-1, 18, 19, 19])

        # Input convolution
        flow = self.conv_block(x_planes, filter_size=3,
                               input_channels=18,
                               output_channels=self.RESIDUAL_FILTERS)
        # Residual tower
        for _ in range(0, self.RESIDUAL_BLOCKS):
            flow = self.residual_block(flow, self.RESIDUAL_FILTERS)

        # Policy head
        conv_pol = self.conv_block(flow, filter_size=1,
                                   input_channels=self.RESIDUAL_FILTERS,
                                   output_channels=2)
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, 2*19*19])
        W_fc1 = weight_variable([2 * 19 * 19, (19 * 19) + 1])
        b_fc1 = bias_variable([(19 * 19) + 1])
        self.weights.append(W_fc1)
        self.weights.append(b_fc1)
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1)

        # Value head
        conv_val = self.conv_block(flow, filter_size=1,
                                   input_channels=self.RESIDUAL_FILTERS,
                                   output_channels=1)
        h_conv_val_flat = tf.reshape(conv_val, [-1, 19*19])
        W_fc2 = weight_variable([19 * 19, 256])
        b_fc2 = bias_variable([256])
        self.weights.append(W_fc2)
        self.weights.append(b_fc2)
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        W_fc3 = weight_variable([256, 1])
        b_fc3 = bias_variable([1])
        self.weights.append(W_fc3)
        self.weights.append(b_fc3)
        h_fc3 = tf.nn.tanh(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))

        return h_fc1, h_fc3

    def snap_save(self):
        # Save a snapshot of all the variables in the current graph.
        if not hasattr(self, 'save_op'):
            save_ops = []
            rest_ops = []
            for var in self.weights:
                if isinstance(var, str):
                    var = tf.get_default_graph().get_tensor_by_name(var)
                name = var.name.split(':')[0]
                v = tf.Variable(var, name='save/'+name, trainable=False)
                save_ops.append(tf.assign(v, var))
                rest_ops.append(tf.assign(var, v))
            self.save_op = tf.group(*save_ops)
            self.restore_op = tf.group(*rest_ops)
        self.session.run(self.save_op)

    def snap_restore(self):
        # Restore variables in the current graph from the snapshot.
        self.session.run(self.restore_op)

    def save_swa_network(self, steps, path, leela_path):
        # Sample 1 in self.swa_c of the networks. Compute in this way so
        # that it's safe to change the value of self.swa_c
        rem = self.session.run(tf.assign_add(self.swa_skip, -1))
        if rem > 0:
            return
        self.swa_skip.load(self.swa_c, self.session)

        # Add the current weight vars to the running average.
        num = self.session.run(self.swa_accum_op)

        if self.swa_max_n != None:
            num = min(num, self.swa_max_n)
            self.swa_count.load(float(num), self.session)

        swa_path = path + "-swa-" + str(int(num)) + "-" + str(steps) + ".txt"

        # save the current network.
        self.snap_save()
        # Copy the swa weights into the current network.
        self.session.run(self.swa_load_op)
        if self.swa_recalc_bn:
            print("Refining SWA batch normalization")
            for _ in range(200):
                self.session.run(
                    [self.loss, self.update_ops, self.next_batch],
                    feed_dict={self.training: True, self.handle: self.train_handle})

        self.save_leelaz_weights(swa_path)
        # restore the saved network.
        self.snap_restore()

        print("Wrote averaged network to {}".format(swa_path))
