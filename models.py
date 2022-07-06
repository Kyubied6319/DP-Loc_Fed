import tensorflow.compat.v1 as tf
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from utils import *

cnf = load_cfg("cfg/cfg_general.json")


def create_model_nh_lstm(max_words_num, out_size):
    inputs = tf.keras.Input(shape=(cnf.GRAM_SIZE, 3))
    split = tf.split(inputs, num_or_size_splits=3, axis=2)

    next_loc = tf.reshape(split[0], [-1, cnf.GRAM_SIZE])
    dst = tf.reshape(split[1], [-1, cnf.GRAM_SIZE])
    time = tf.reshape(split[2], [-1, cnf.GRAM_SIZE, 1])

    emb_layer = tf.keras.layers.Embedding(max_words_num + 1, cnf.NH_EMBEDDING_SIZE, input_length=cnf.GRAM_SIZE)

    merge = tf.keras.layers.concatenate([emb_layer(next_loc), emb_layer(dst), time], axis=2)

    lstm = tf.keras.layers.LSTM(cnf.NH_HIDDEN_SIZE)(merge)

    # next-hop: predict only neighbors
    next_pred = tf.keras.layers.Dense(out_size)(lstm)

    return tf.keras.Model(inputs=inputs, outputs=next_pred)


def create_model_nh_ffn(max_words_num, out_size):
    inputs = tf.keras.Input(shape=(cnf.GRAM_SIZE, 3))
    split = tf.split(inputs, num_or_size_splits=3, axis=2)

    next_loc = tf.reshape(split[0], [-1, cnf.GRAM_SIZE])
    dst = tf.reshape(split[1], [-1, cnf.GRAM_SIZE])
    time = tf.reshape(split[2], [-1, cnf.GRAM_SIZE, 1])

    emb_layer = tf.keras.layers.Embedding(max_words_num + 1, cnf.NH_EMBEDDING_SIZE, input_length=cnf.GRAM_SIZE)

    merge = tf.keras.layers.concatenate([emb_layer(next_loc), emb_layer(dst), time], axis=2)
    # merge = tf.keras.layers.concatenate([emb_layer(next_loc), time], axis=2)

    flatten = tf.keras.layers.Flatten()(merge)

    dense = tf.keras.layers.Dense(cnf.NH_HIDDEN_SIZE, activation='relu')(flatten)

    # next-hop: predict only neighbors
    next_pred = tf.keras.layers.Dense(out_size)(dense)

    return tf.keras.Model(inputs=inputs, outputs=next_pred)


class VAE(tf.keras.Model):
    """Variational autoencoder."""

    def __init__(self, original_dim, latent_dim, hidden_dim, max_words_num, max_slots):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.max_slots = max_slots
        self.max_word_num = max_words_num

        # Define encoder model.
        inp_e = tf.keras.layers.Input(shape=(original_dim,), name="encoder_input")
        split = tf.split(inp_e, num_or_size_splits=3, axis=1)

        src, dst, time = split[0], split[1], split[2]
        src = tf.reshape(split[0], [-1])
        dst = tf.reshape(split[1], [-1])
        time = tf.reshape(split[2], [-1])

        src = tf.one_hot(tf.cast(src, dtype=tf.int32), depth=max_words_num)
        dst = tf.one_hot(tf.cast(dst, dtype=tf.int32), depth=max_words_num)
        time = tf.one_hot(tf.cast(time, dtype=tf.int32), depth=max_slots)

        merge = tf.keras.layers.concatenate([src, dst, time], axis=1)

        hd_d = tf.keras.layers.Dense(units=hidden_dim, activation=tf.nn.relu)(merge)
        # No activation
        latent = tf.keras.layers.Dense(latent_dim + latent_dim)(hd_d)
        # for AE:
        # tf.keras.layers.Dense(latent_dim)

        self.encoder = tf.keras.Model(inputs=inp_e, outputs=latent, name="Encoder")

        # Define decoder model.
        inp_d = tf.keras.layers.Input(shape=(latent_dim,), name="decoder_input")
        hd_d = tf.keras.layers.Dense(units=hidden_dim, activation=tf.nn.relu)(inp_d)

        src_out = tf.keras.layers.Dense(units=max_words_num)(hd_d)
        dst_out = tf.keras.layers.Dense(units=max_words_num)(hd_d)
        time_out = tf.keras.layers.Dense(units=max_slots)(hd_d)

        self.decoder = tf.keras.Model(inputs=inp_d, outputs=[src_out, dst_out, time_out], name="Decoder")

    def call(self, x):
        mean, logvar = self.encode(x)
        # MCMC estimate with a single sample
        z = self.reparameterize(mean, logvar)
        src, dst, ts = self.decode(z, apply_softmax=True)
        src = tf.math.argmax(src, axis=1, output_type=tf.int32)
        dst = tf.math.argmax(dst, axis=1, output_type=tf.int32)
        ts = tf.math.argmax(ts, axis=1, output_type=tf.int32)

        return tf.stack([src, dst, ts], axis=1)

    @tf.function
    def sample(self, num, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(num, self.latent_dim))

        src, dst, ts = self.decode(eps, apply_softmax=True)
        src = tf.math.argmax(src, axis=1, output_type=tf.int32)
        dst = tf.math.argmax(dst, axis=1, output_type=tf.int32)
        ts = tf.math.argmax(ts, axis=1, output_type=tf.int32)

        return tf.stack([src, dst, ts], axis=1)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_softmax=True):
        src, dst, ts = self.decoder(z)
        if apply_softmax:
            src = tf.nn.softmax(src)
            dst = tf.nn.softmax(dst)
            ts = tf.nn.softmax(ts)

        return [src, dst, ts]
