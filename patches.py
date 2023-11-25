import tensorflow as tf
from tensorflow.keras import layers

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({"patch_size": self.patch_size})
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.class_token = tf.Variable(
            initial_value=tf.zeros_initializer()(
                shape=(1, 1, projection_dim), dtype="float32"
            ),
            trainable=True,
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({"projection_dim": self.projection_dim})
        return config

    def call(self, patch):
        batch_size = tf.shape(patch)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.class_token, [batch_size, 1, self.projection_dim]),
            dtype=patch.dtype,
        )
        return tf.concat([cls_broadcasted, self.projection(patch)], 1)