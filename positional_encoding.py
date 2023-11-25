import tensorflow as tf
from tensorflow.keras import layers

class PositionalEmbedding(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PositionalEmbedding, self).__init__()
        self.projection_dim = projection_dim
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim = num_patches + 1, output_dim = projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {"projection_dim": self.projection_dim, "num_patches": self.num_patches}
        )
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        batch_size = tf.shape(patch)[0]
        embs = tf.cast(
            tf.broadcast_to(
                self.position_embedding(positions),
                [batch_size, self.num_patches + 1, self.projection_dim],
            ),
            dtype=patch.dtype,
        )
        return patch + embs