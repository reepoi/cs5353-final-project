import logging

import tensorflow as tf
from confugue import configurable
from museflow.components import Component, using_scope

_LOGGER = logging.getLogger(__name__)


@configurable
class CNNVAE(Component):

    def __init__(self, training=None, name='cnnvae'):
        Component.__init__(self, name=name)

        self._is_training = training
        with self.use_scope():
            self._layers_2d = self._cfg['2d_layers'].configure_list()
            import pdb
            pdb.set_trace()
            self._layers_1d = self._cfg['1d_layers'].configure_list()

        self.magic_prime = 19

        # [batch_size, time, channels] -> [batch_size, time, magic_prime * 2]
        # self.statistics = tf.keras.Sequential([tf.keras.layers.Dense(units=self.magic_prime * 2)])
        self.statistics = tf.keras.layers.Dense(units=self.magic_prime * 2)

        # [batch_size, time, magic_prime] -> [batch_size, time, num_channels]
        # self.samplings = tf.keras.Sequential([tf.keras.layers.Dense(units=1024)])
        self.samplings = tf.keras.layers.Dense(units=1024)

    def __call__(self, inputs):
        return self.apply(inputs)

    @using_scope
    def apply(self, inputs):
        batch_size = tf.shape(inputs)[0]
        features = inputs
        if self._layers_2d:
            if features.shape.ndims == 3:
                # Expand to 4 dimensions: [batch_size, rows, time, channels]
                features = tf.expand_dims(features, -1)

            # 2D layers: 4 -> 4 dimensions
            for layer in self._layers_2d:
                _LOGGER.debug(f'Inputs to layer {layer} have shape {features.shape}')
                features = self._apply_layer(layer, features)
            _LOGGER.debug(f'After the 2D layers, the features have shape {features.shape}')

            # Features have shape [batch_size, rows, time, channels]. Switch rows and cols, then
            # flatten rows and channels to get 3 dimensions: [batch_size, time, new_channels].
            features = tf.transpose(features, perm=[0, 2, 1, *range(3, features.shape.ndims)])
            num_channels = features.shape[2] * features.shape[3]
            features = tf.reshape(features, [batch_size, -1, num_channels])

        # 1D layers: 3 -> 3 dimensions: [batch_size, time, channels]
        for layer in self._layers_1d:
            _LOGGER.debug(f'Inputs to layer {layer} have shape {features.shape}')
            features = self._apply_layer(layer, features)

        import pdb
        pdb.set_trace()

        # [batch_size, time, magic_prime * 2] -> [batch_size, time, 2, magic_prime]
        mean, logvar = tf.split(self.statistics(features), num_or_size_splits=2, axis=-1)

        # [batch_size, time, magic_prime]

        eps = tf.random.normal(shape=tf.shape(mean))

        # [batch_size, time, magic_prime]
        reparameterize = eps * tf.exp(logvar * 0.5) + mean

        # [batch_size, time, magic_prime] -> [batch_size, time, num_channels]
        features = self.samplings(reparameterize)

        return features, (mean, logvar), reparameterize


    def _apply_layer(self, layer, features):
        if isinstance(layer, (tf.layers.Dropout, tf.keras.layers.Dropout)):
            return layer(features, training=self._is_training)
        return layer(features)


@configurable
class CNN(Component):

    def __init__(self, training=None, name='cnn'):
        Component.__init__(self, name=name)

        self._is_training = training
        with self.use_scope():
            self._layers_2d = self._cfg['2d_layers'].configure_list()
            self._layers_1d = self._cfg['1d_layers'].configure_list()

    def __call__(self, inputs):
        return self.apply(inputs)

    @using_scope
    def apply(self, inputs):
        batch_size = tf.shape(inputs)[0]
        features = inputs
        if self._layers_2d:
            if features.shape.ndims == 3:
                # Expand to 4 dimensions: [batch_size, rows, time, channels]
                features = tf.expand_dims(features, -1)

            # 2D layers: 4 -> 4 dimensions
            for layer in self._layers_2d:
                _LOGGER.debug(f'Inputs to layer {layer} have shape {features.shape}')
                features = self._apply_layer(layer, features)
            _LOGGER.debug(f'After the 2D layers, the features have shape {features.shape}')

            # Features have shape [batch_size, rows, time, channels]. Switch rows and cols, then
            # flatten rows and channels to get 3 dimensions: [batch_size, time, new_channels].
            features = tf.transpose(features, perm=[0, 2, 1, *range(3, features.shape.ndims)])
            num_channels = features.shape[2] * features.shape[3]
            features = tf.reshape(features, [batch_size, -1, num_channels])

        # 1D layers: 3 -> 3 dimensions: [batch_size, time, channels]
        for layer in self._layers_1d:
            _LOGGER.debug(f'Inputs to layer {layer} have shape {features.shape}')
            features = self._apply_layer(layer, features)

        return features

    def _apply_layer(self, layer, features):
        if isinstance(layer, (tf.layers.Dropout, tf.keras.layers.Dropout)):
            return layer(features, training=self._is_training)
        return layer(features)
