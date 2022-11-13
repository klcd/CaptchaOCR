import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class CTCLayer(layers.Layer):
    '''
        CTC scoring layer. For more information see here

    '''


    def __init__(self, name=None, **kwargs):
        self.loss_fn = keras.backend.ctc_batch_cost
        super(CTCLayer, self).__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred
