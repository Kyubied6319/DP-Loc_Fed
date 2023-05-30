# Copyright 2018, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training with differentially private optimizers (tensorflow)."""

import random
import sys
import flwr as fl
import argparse
import os

import tensorflow.compat.v1 as tf

# our modules
from models import *
from preproc import *
from privacy_accountant import MomentsAccountant
from train_step import *
from utils import *
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
cnf = load_cfg("cfg/cfg_general.json")
cnf.CELL_SIZE = int(float(sys.argv[2]))
cnf.__dict__.update(load_cfg(sys.argv[3]).__dict__)

# tf.disable_v2_behavior()
# disable_eager_execution()

# we return a loss per sample in a batch (and not a single aggregated loss value of the whole batch)
loss_object_nh = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


# DP sampling
def batch_generator(data, labels, batch_size, sample_locs):
    # First sample a trace and then a sample
    # we drop samples if len(data) % batch_size != 0
    for _ in range(len(data) // batch_size):
        pos = [random.choice(random.choice(sample_locs)) for _ in range(batch_size)]
        yield tf.convert_to_tensor(data[pos]), tf.convert_to_tensor(labels[pos])

    # For next-hop model


def compute_nh_loss(model, x, y):
    return loss_object_nh(y_true=y, y_pred=model(x, training=True))


# For VAE
def compute_vae_loss(model, x, y):
    mean, logvar = model.encode(x)
    # MCMC estimate with a single sample
    z = model.reparameterize(mean, logvar)
    src_pred, dst_pred, ts_pred = model.decode(z, apply_softmax=False)

    rec_loss_1 = tf.keras.losses.sparse_categorical_crossentropy(y[:, 0], src_pred, from_logits=True)
    rec_loss_2 = tf.keras.losses.sparse_categorical_crossentropy(y[:, 1], dst_pred, from_logits=True)
    rec_loss_3 = tf.keras.losses.sparse_categorical_crossentropy(y[:, 2], ts_pred, from_logits=True)
    rec_loss = (rec_loss_1 + rec_loss_2 + rec_loss_3) / 3

    kl_loss = - 0.5 * tf.reduce_mean(logvar - tf.square(mean) - tf.exp(logvar) + 1)
    # tf.print("rec_loss:", rec_loss, "kl_loss:", kl_loss)
    return tf.reduce_mean(rec_loss + kl_loss)


# this is the training procedure (DP and NON-DP too) for the init model
def train_init(model, train_data, sample_locs, batch_size, epochs):
    log_name = "init"

    print(model.encoder.summary())
    print(model.decoder.summary())

    optimizer_init = tf.keras.optimizers.SGD(learning_rate=cnf.LEARNING_RATE_VAE_DP)

    train_size = len(train_data)
    trace_num = len(sample_locs)

    batch_num = train_size // batch_size

    # q = 1 - (1.0 - BATCH_SIZE / train_size) ** max_len
    q = batch_size / trace_num
    print("=> with Differential Privacy (clipping : %.2f, noise scale: %.2f, sampling prob: %.3f)" % (
        cnf.L2_NORM_CLIP_VAE, cnf.SIGMA_VAE * cnf.L2_NORM_CLIP_VAE, q))
    print("=> Learning rate:", cnf.LEARNING_RATE_VAE_DP)
    print("=> Batch size:", batch_size)
    print("=> Epochs:", epochs)

    # Keep results for plotting
    train_loss_results = []

    # keep track of privacy budget
    priv_acc = MomentsAccountant(delta=1 / trace_num, sigma=cnf.SIGMA_VAE, sampling_prob=q)

    # Training loop
    iters = 0
    for epoch in range(epochs):

        epoch_loss_avg = tf.keras.metrics.Mean()
        #progbar = tf.keras.utils.Progbar(batch_num)
        # x=y now (generative model)
        for eps_value, (x, y) in priv_acc.make_iter(batch_generator(train_data, train_data, batch_size, sample_locs)):
            loss_values, norm_values = train_step_DP(model, compute_vae_loss, optimizer_init, x, y,
                                                     tf.constant(cnf.L2_NORM_CLIP_VAE, dtype=tf.float32),
                                                     tf.constant(cnf.SIGMA_VAE, dtype=tf.float32))

            # Track progress
            loss_value = tf.reduce_mean(input_tensor=loss_values)
            norm_value = tf.reduce_mean(input_tensor=norm_values)

            # Add current batch loss
            epoch_loss_avg.update_state(loss_value)

            iters += 1

            #progbar.add(1, values=[('loss', loss_value), ('norm', norm_value), ('eps', eps_value)])

        # Train performance
        train_loss_results.append(epoch_loss_avg.result())
        print(
            "({:s}) Iterations: {:03d}, Epoch: {:03d}, Loss: {:.3f}, Epsilon (privacy): {:.3f}".format(
                log_name,
                iters,
                epoch + 1,
                epoch_loss_avg.result(),
                eps_value
            )
        )
        rep_file = cnf.VAE_TRAIN_METRICS_FILE % (cnf.CELL_SIZE, cnf.EPS)
        pickle.dump(train_loss_results, open(rep_file, "wb"))


# this is the training procedure (DP and NON-DP too) for next-hop model
def train_nh(model, train_data, train_labels, test_data, test_labels, sample_locs, neighbor_err, batch_size, epochs):
    log_name = "next_hop"

    print(model.summary())

    optimizer_nh = tf.keras.optimizers.SGD(learning_rate=cnf.LEARNING_RATE_NH_DP)

    train_size = len(train_data)
    trace_num = len(sample_locs)

    batch_num = train_size // batch_size

    # train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size)

    q = batch_size / trace_num
    print("=> with Differential Privacy (clipping : %.2f, noise scale: %.2f, sampling prob: %.3f)" % (
        cnf.L2_NORM_CLIP_NH, cnf.SIGMA_NH * cnf.L2_NORM_CLIP_NH, q))
    print("=> Learning rate:", cnf.LEARNING_RATE_NH_DP)
    print("=> Batch size:", batch_size)
    print("=> Epochs:", epochs)

    # keep track of privacy budget
    priv_acc = MomentsAccountant(delta=1 / trace_num, sigma=cnf.SIGMA_NH, sampling_prob=q)

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    train_error_results = []

    @tf.function
    def dist_error(y, y_pred, errors):
        y = tf.cast(y, dtype=tf.int32)

        # in tf.function map_fn is parallelized
        err = tf.map_fn(lambda x: errors[x[0], x[1]], (y_pred, y), tf.float32)

        return tf.reduce_mean(err)

    neighbor_err = tf.convert_to_tensor(neighbor_err, dtype=tf.float32)

    iters = 0

    for epoch in range(epochs):

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_err_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.Accuracy()
        #progbar = tf.keras.utils.Progbar(batch_num)

        # Training loop
        # for i, (x, y) in enumerate(train_dataset):
        for eps_value, (x, y) in priv_acc.make_iter(batch_generator(train_data, train_labels, batch_size, sample_locs)):
            iters += 1

            loss_values, norm_values = train_step_DP(model, compute_nh_loss, optimizer_nh, x, y,
                                                     tf.constant(cnf.L2_NORM_CLIP_NH, dtype=tf.float32),
                                                     tf.constant(cnf.SIGMA_NH, dtype=tf.float32))

            # Track progress
            y_pred = tf.argmax(model(x, training=True), axis=1, output_type=tf.int32)

            loss_value = tf.reduce_mean(input_tensor=loss_values)
            norm_value = tf.reduce_mean(input_tensor=norm_values)

            err_value = dist_error(y, y_pred, neighbor_err)
            max_occ_pred = tf.math.reduce_max(tf.math.bincount(y_pred))
            max_occ = tf.math.reduce_max(tf.math.bincount(tf.cast(y, dtype=tf.int32)))

            epoch_loss_avg.update_state(loss_value)
            epoch_err_avg.update_state(err_value)
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y, y_pred)
            vals = [('loss', loss_value), ('norm', norm_value), ('accuracy', epoch_accuracy.result()),
                    ('err', err_value), ('eps', eps_value), ('occ', max_occ), ('occ_pred', max_occ_pred)]

            #progbar.add(1, values=vals)

        # End epoch

        # Train performance
        print(
            "({:s}) Iterations: {:03d}, Epoch: {:03d}, Loss: {:.3f}, Accuracy: {:.3%}, Error: {:.3f} meters, Epsilon (privacy): {:.3f}".format(
                log_name,
                iters, epoch + 1, epoch_loss_avg.result(), epoch_accuracy.result(), epoch_err_avg.result(), eps_value
            )
        )

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        train_error_results.append(epoch_err_avg.result())

        # Test performance
        test_accuracy = tf.keras.metrics.Accuracy()
        test_error = tf.keras.metrics.Mean()
        for (x, y) in test_dataset:
            # training=False is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            logits = model(x, training=False)
            prediction = tf.argmax(logits, axis=1, output_type=tf.int32)

            test_accuracy(prediction, y)
            test_error(dist_error(y, prediction, neighbor_err))

        print("({:s}) Test set accuracy: {:.3%}, error: {:4f} meters".format(log_name, test_accuracy.result(),
                                                                             test_error.result(), test_error.result()))

        rep_file = cnf.NH_TRAIN_METRICS_FILE % (cnf.CELL_SIZE, cnf.EPS)
        pickle.dump([train_loss_results, train_accuracy_results, train_error_results], open(rep_file, "wb"))

def main() -> None:
    # tf.logging.set_verbosity(tf.logging.ERROR)

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    # Preprocessing
    preproc = Preprocessing(cnf.CELL_SIZE, cnf.EPS)
    
    #Data
    train_data, train_labels, test_data, test_labels, init_train_data, sample_locs_nh, sample_locs_init = preproc.load_data(
    cnf.SIGMA_TK, train_size=0.9)

    # Plotting original data
    orig = preproc.convert_init_data_to_coords(init_train_data)
    preproc.plot_init_data("orig", orig)

    #Data split for two devices
    if sys.argv[4] == True:
        train_data, train_labels, test_data, test_labels = train_data[:(len(train_data)//2)], train_labels[:len(train_labels)//2], test_data[:len(test_data)//2], test_labels[:len(test_labels)//2] 
        init_train_data, sample_locs_nh, sample_locs_init = init_train_data[:len(init_train_data)//2], sample_locs_nh[:len(sample_locs_nh)//2], sample_locs_init[:len(sample_locs_init)//2]
    else:
        train_data, train_labels, test_data, test_labels = train_data[(len(train_data)//2):], train_labels[len(train_labels)//2:], test_data[len(test_data)//2:], test_labels[len(test_labels)//2:] 
        init_train_data, sample_locs_nh, sample_locs_init = init_train_data[len(init_train_data)//2:], sample_locs_nh[len(sample_locs_nh)//2:], sample_locs_init[len(sample_locs_init)//2:]
    #Training clients
    if sys.argv[1] == "VAE":
        print("=== Training init model...1")
        vae = VAE(original_dim=cnf.VAE_ORIG_DIM, latent_dim=cnf.VAE_LATENT_DIM, hidden_dim=cnf.VAE_HIDDEN_DIM,
                  max_words_num=len(preproc.cell2token), max_slots=preproc.MAX_SLOTS)
        
        class Client(fl.client.NumPyClient):
            
            def get_parameters(self, config):
                return vae.get_weights()

            def fit(self, parameters, config):
                vae.set_weights(parameters)
                train_init(vae, init_train_data, sample_locs_init, batch_size=cnf.BATCH_SIZE_VAE_DP, epochs=cnf.EPOCHS_VAE_DP)
                return vae.get_weights(), len(self.train_data), {}

            def evaluate(self, parameters,config):
                vae.set_weights(parameters)
                loss = vae.model.compute_vae_loss(test_data,test_labels)
                return loss
        
        fl.client.start_numpy_client(
            server_address="127.0.0.1:"+"33432", 
            client=Client(), 
        )

    elif sys.argv[1] == "TRACES":
        # Train next_hop model
        print("=== Training next-hop model...2")
        if cnf.PREPROC_MAP_TO_TOP_K:
            model_next_hop = create_model_nh_ffn(len(preproc.cell2token), preproc.top_k_size)
            
        else:
            model_next_hop = create_model_nh_ffn(len(preproc.cell2token), preproc.nl_size)
            
        class Client(fl.client.NumPyClient):
            def get_parameters(self,config):
                return model_next_hop.get_weights()

            def fit(self, parameters, config):
                model_next_hop.set_weights(parameters)
                if cnf.PREPROC_MAP_TO_TOP_K:
                    train_nh(model_next_hop, train_data, train_labels, test_data, test_labels, sample_locs_nh,
                     preproc.top_k_err, batch_size=cnf.BATCH_SIZE_NH_DP, epochs=cnf.EPOCHS_NH_DP)
                else:
                    train_nh(model_next_hop, train_data, train_labels, test_data, test_labels, sample_locs_nh,
                            preproc.neighbor_err, batch_size=cnf.BATCH_SIZE_NH_DP, epochs=cnf.EPOCHS_NH_DP)
                return model_next_hop.get_weights(), len(self.train_data), {}

            def evaluate(self, parameters,config):
                model_next_hop.set_weights(parameters)
                loss = model_next_hop.model.compute_nh_loss(test_data,test_labels)
                return loss
            
        fl.client.start_numpy_client(
            server_address="127.0.0.1:9001",
            client=Client(), 
        )
        #rep_file = cnf.PATH_NH % (cnf.CELL_SIZE, cnf.EPS)
        #print("Saving model to %s..." % rep_file)
        #model_next_hop.save_weights(rep_file)

    else:
        # Train init model
        print("=== Training init model...3")
        vae = VAE(original_dim=cnf.VAE_ORIG_DIM, latent_dim=cnf.VAE_LATENT_DIM, hidden_dim=cnf.VAE_HIDDEN_DIM,
                  max_words_num=len(preproc.cell2token), max_slots=preproc.MAX_SLOTS)
        train_init(vae, init_train_data, sample_locs_init, batch_size=cnf.BATCH_SIZE_VAE_DP, epochs=cnf.EPOCHS_VAE_DP)

        # doesn't work: vae.save(PATH_VAE)
        rep_file = cnf.PATH_VAE % (cnf.CELL_SIZE, cnf.EPS)
        print("Saving model to %s..." % rep_file)
        vae.save_weights(rep_file)
        # tf.saved_model.save(vae, "saved_model/vae")

        # Plotting original data
        orig = preproc.convert_init_data_to_coords(init_train_data)
        preproc.plot_init_data("orig", orig)
        

        # Train next_hop model
        print("=== Training next-hop model...4")
        if cnf.PREPROC_MAP_TO_TOP_K:
            print("preproc.topk_size:  ", preproc.top_k_size)
            print("preproc.cell2token: ", preproc.cell2token)
            model_next_hop = create_model_nh_ffn(len(preproc.cell2token), preproc.top_k_size)
            train_nh(model_next_hop, train_data, train_labels, test_data, test_labels, sample_locs_nh,
                     preproc.top_k_err, batch_size=cnf.BATCH_SIZE_NH_DP, epochs=cnf.EPOCHS_NH_DP)
        else:
            model_next_hop = create_model_nh_ffn(len(preproc.cell2token), preproc.nl_size)
            train_nh(model_next_hop, train_data, train_labels, test_data, test_labels, sample_locs_nh,
                     preproc.neighbor_err, batch_size=cnf.BATCH_SIZE_NH_DP, epochs=cnf.EPOCHS_NH_DP)

        rep_file = cnf.PATH_NH % (cnf.CELL_SIZE, cnf.EPS)
        print("Saving model to %s..." % rep_file)
        model_next_hop.save_weights(rep_file)



if __name__ == '__main__':
    main()
