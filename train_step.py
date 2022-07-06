import tensorflow.compat.v1 as tf
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

@tf.function
def train_step_DP(model, loss_object, optimizer, x, y, l2norm_clip, sigma):
    # loss_values = loss_object(y_true=y, y_pred=model(x, training=True))

    p_batch_size = tf.shape(input=x)[0]
    var_list = model.trainable_variables

    def zeros_like(arg):
        """A `zeros_like` function that also works for `tf.TensorSpec`s."""
        try:
            arg = tf.convert_to_tensor(value=arg)
        except TypeError:
            pass
        return tf.zeros(arg.shape, arg.dtype)

    sum_grads = tf.nest.map_structure(zeros_like, var_list)

    loss_values = tf.TensorArray(dtype=tf.float32, size=p_batch_size)
    glob_norms = tf.TensorArray(dtype=tf.float32, size=p_batch_size)

    for idx in tf.range(p_batch_size):
        # Compute gradient per sample (record)
        with tf.GradientTape() as g:
            x_rec = tf.expand_dims(tf.gather(x, idx), 0)
            y_rec = tf.expand_dims(tf.gather(y, idx), 0)
            rec_loss = loss_object(model, x_rec, y_rec)

        loss_values = loss_values.write(idx, rec_loss)

        grads = g.gradient(rec_loss, var_list)

        # Clip
        # TODO: Check for VAE whether all params are considered for noising!!!!
        record_as_list = tf.nest.flatten(grads)
        clipped_as_list, glob_norm = tf.clip_by_global_norm(record_as_list, l2norm_clip)
        glob_norms = glob_norms.write(idx, glob_norm)
        grads = tf.nest.pack_sequence_as(grads, clipped_as_list)

        sum_grads = tf.nest.map_structure(tf.add, sum_grads, grads)

    # Add noise
    random_normal = tf.random_normal_initializer(stddev=l2norm_clip * sigma)

    def add_noise(v):
        return v + random_normal(tf.shape(input=v))

    sum_grads = tf.nest.map_structure(add_noise, sum_grads)

    # Take the average over all samples
    def normalize(v):
        return tf.truediv(v, tf.cast(p_batch_size, tf.float32))

    final_grads = tf.nest.map_structure(normalize, sum_grads)

    # loss_v = tf.reduce_mean(loss_values)
    # tf.print(tf.reduce_mean(loss_values.stack()))
    optimizer.apply_gradients(zip(final_grads, var_list))
    return loss_values.stack(), glob_norms.stack()


@tf.function
def train_step_NODP(model, loss_object, optimizer, x, y):
    var_list = model.trainable_variables

    with tf.GradientTape() as g:
        loss_values = loss_object(y_true=y, y_pred=model(x, training=True))
        loss_val = tf.reduce_mean(loss_values)

    grads = g.gradient(loss_val, var_list)

    optimizer.apply_gradients(zip(grads, var_list))

    return loss_values, tf.constant([0.0], dtype=tf.float32)
