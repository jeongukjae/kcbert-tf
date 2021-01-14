import json

import tensorflow as tf


def gelu(x):
    return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * x * (1 + 0.044715 * x * x)))


class BertConfig:
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 300,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        **kwargs,  # unused
    ):
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = gelu
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

    @staticmethod
    def from_json(path: str) -> "BertConfig":
        with open(path, "r") as f:
            file_content = json.load(f)

        return BertConfig(**file_content)

class BertModel(tf.keras.Model):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        # embedding layer
        self.token_embeddings = tf.keras.layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name="token_embeddings",
        )
        self.token_type_embeddings = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name="token_type_embeddings",
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name="position_embeddings",
        )
        self.embedding_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, axis=-1, name="embedding_layer_norm", dtype='float32'
        )

        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

        # encoder
        self.encoders = [
            TransformerEncoder(config, name=f"encoder_{i}")
            for i in range(config.num_hidden_layers)
        ]

        # pooler
        self.pooler_layer = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            activation="tanh",
            name="pooler_layer",
        )

    def call(self, inputs):
        input_ids = inputs['input_word_ids']
        attention_mask = tf.cast(inputs['input_mask'], self.dtype)
        token_type_ids = inputs['input_type_ids']

        with tf.name_scope("embedding"):
            position_ids = tf.range(tf.shape(input_ids)[-1], dtype=input_ids.dtype)[tf.newaxis, :]
            words_embeddings = self.token_embeddings(input_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            position_embeddings = self.position_embeddings(position_ids)

            embeddings = words_embeddings + token_type_embeddings + position_embeddings
            embeddings = self.embedding_layer_norm(embeddings)
            hidden_states = self.dropout(embeddings)

        attention_mask = (1.0 - attention_mask[:, tf.newaxis, tf.newaxis, :]) * -10000

        for encoder in self.encoders:
            hidden_states = encoder(hidden_states, attention_mask)

        pooled_output = self.pooler_layer(hidden_states[:, 0, :])

        return {
            "sequence_output": hidden_states,
            "pooled_output": pooled_output,
        }


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.qkv = tf.keras.layers.Dense(
            config.hidden_size * 3,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name="qkv",
        )
        self.attention_dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name="attention_dense",
        )
        self.attention_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps,
            axis=-1,
            name="attention_layer_norm",
            dtype='float32',
        )

        self.intermediate_dense = tf.keras.layers.Dense(
            config.intermediate_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            activation=config.hidden_act,
            name="intermediate_dense",
        )
        self.intermediate_dense2 = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name="intermediate_dense2",
        )
        self.intermediate_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps,
            axis=-1,
            name="intermediate_layer_norm",
            dtype='float32'
        )

        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

        self.num_head = config.num_attention_heads
        self.head_size = int(config.hidden_size / self.num_head)
        self.scaling_factor = float(self.head_size) ** -0.5
        self.hidden_size = config.hidden_size

    def call(self, sequence, mask=None):
        # multihead attention
        attention = self._multihead_attention(sequence, mask)
        # add and norm
        attention = self.attention_layer_norm(attention + sequence)
        # fc
        with tf.name_scope('intermediate'):
            intermediate = self.intermediate_dense(attention)
            intermediate = self.dropout(self.intermediate_dense2(intermediate))
        # add and norm
        intermediate = self.intermediate_layer_norm(intermediate + attention)
        return intermediate

    def _multihead_attention(self, sequence, mask):
        with tf.name_scope("multihead_attention"):
            q, k, v = tf.split(self.qkv(sequence), num_or_size_splits=3, axis=-1)

            with tf.name_scope("attention"):
                q = self._reshape_qkv(q)
                k = self._reshape_qkv(k)
                v = self._reshape_qkv(v)

                # calculate attention
                attention = tf.matmul(q, k, transpose_b=True)
                attention *= self.scaling_factor
                if mask is not None:
                    attention += mask
                attention = tf.keras.layers.Softmax(axis=-1, dtype='float32')(attention)
                attention = self.dropout(attention)
                attention = tf.matmul(attention, v)

                # concat
                attention = tf.transpose(attention, perm=[0, 2, 1, 3])
                attention = tf.reshape(attention, [-1, tf.shape(attention)[1], self.hidden_size])

            # last dense net
            attention = self.attention_dense(attention)
            attention = self.dropout(attention)
            return attention

    def _reshape_qkv(self, val):
        new_shape = [-1, tf.shape(val)[1], self.num_head, self.head_size]
        return tf.transpose(tf.reshape(val, new_shape), perm=[0, 2, 1, 3])
