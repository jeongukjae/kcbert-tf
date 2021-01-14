import os

import numpy as np
import tensorflow as tf
import tensorflow_text as text
import torch
from transformers import AutoTokenizer, BertModel as TorchBertModel
from model import BertConfig, BertModel

tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
torch_model = TorchBertModel.from_pretrained("beomi/kcbert-base").eval()

if not os.path.isdir("kcbert-base"):
    os.mkdir("kcbert-base")

tokenizer.save_vocabulary("kcbert-base")

config = BertConfig(vocab_size=30000)
model = BertModel(config)
model(
    {
        "input_word_ids": tf.keras.Input(shape=[None], dtype=tf.int64),
        "input_mask": tf.keras.Input(shape=[None], dtype=tf.int64),
        "input_type_ids": tf.keras.Input(shape=[None], dtype=tf.int64),
    }
)

sd = torch_model.state_dict()
model.token_embeddings.set_weights([sd["embeddings.word_embeddings.weight"].numpy()])
model.position_embeddings.set_weights([sd["embeddings.position_embeddings.weight"].numpy()])
model.token_type_embeddings.set_weights([sd["embeddings.token_type_embeddings.weight"].numpy()])
model.embedding_layer_norm.set_weights([sd["embeddings.LayerNorm.weight"], sd["embeddings.LayerNorm.bias"]])

for i in range(config.num_hidden_layers):
    qkv_weight = np.concatenate([
        sd[f"encoder.layer.{i}.attention.self.query.weight"].T,
        sd[f"encoder.layer.{i}.attention.self.key.weight"].T,
        sd[f"encoder.layer.{i}.attention.self.value.weight"].T,
    ], axis=1)
    qkv_bias = np.concatenate([
        sd[f"encoder.layer.{i}.attention.self.query.bias"],
        sd[f"encoder.layer.{i}.attention.self.key.bias"],
        sd[f"encoder.layer.{i}.attention.self.value.bias"],
    ], axis=0)
    model.encoders[i].qkv.set_weights([qkv_weight, qkv_bias])
    model.encoders[i].attention_dense.set_weights([
        sd[f"encoder.layer.{i}.attention.output.dense.weight"].T,
        sd[f"encoder.layer.{i}.attention.output.dense.bias"],
    ])
    model.encoders[i].attention_layer_norm.set_weights([
        sd[f'encoder.layer.{i}.attention.output.LayerNorm.weight'],
        sd[f'encoder.layer.{i}.attention.output.LayerNorm.bias'],
    ])
    model.encoders[i].intermediate_dense.set_weights([
        sd[f'encoder.layer.{i}.intermediate.dense.weight'].T,
        sd[f'encoder.layer.{i}.intermediate.dense.bias'],
    ])
    model.encoders[i].intermediate_dense2.set_weights([
        sd[f'encoder.layer.{i}.output.dense.weight'].T,
        sd[f'encoder.layer.{i}.output.dense.bias'],
    ])
    model.encoders[i].intermediate_layer_norm.set_weights([
        sd[f'encoder.layer.{i}.output.LayerNorm.weight'],
        sd[f'encoder.layer.{i}.output.LayerNorm.bias'],
    ])

model.pooler_layer.set_weights([
    sd['pooler.dense.weight'].T,
    sd['pooler.dense.bias'],
])
tf.saved_model.save(model, 'kcbert-base/model/0')

to_export = tf.Module()
tokenizer = text.BertTokenizer('./kcbert-base/vocab.txt')
cls_id = 2
sep_id = 3

@tf.function(input_signature=[tf.TensorSpec([None], tf.string), tf.TensorSpec([], tf.int32)])
def call(input_tensor, seq_length):
    batch_size = tf.shape(input_tensor)[0]

    def _parse_single_sentence(x):
        return tf.concat([[cls_id], x[: seq_length - 2], [sep_id]], axis=0)

    tokenized = tokenizer.tokenize(input_tensor)
    tokenized = tokenized.merge_dims(1, 2)
    input_word_ids = tf.map_fn(
        _parse_single_sentence,
        tokenized,
        fn_output_signature=tf.RaggedTensorSpec([None], tf.int64),
    )
    input_mask = tf.ones_like(input_word_ids, dtype=tf.int64)
    input_type_ids = tf.zeros_like(input_word_ids, dtype=tf.int64)

    return {
        "input_word_ids": input_word_ids.to_tensor(shape=[batch_size, seq_length]),
        "input_mask": input_mask.to_tensor(shape=[batch_size, seq_length]),
        "input_type_ids": input_type_ids.to_tensor(shape=[batch_size, seq_length]),
    }

@tf.function(input_signature=[[tf.TensorSpec([None], tf.string), tf.TensorSpec([None], tf.string)], tf.TensorSpec([], tf.int32)])
def call_2(input_tensor, seq_length):
    segment_a = input_tensor[0]
    segment_b = input_tensor[1]

    batch_size = tf.shape(segment_a)[0]

    segment_a = tokenizer.tokenize(segment_a)
    segment_b = tokenizer.tokenize(segment_b)

    segment_a = segment_a.merge_dims(1, 2)
    segment_b = segment_b.merge_dims(1, 2)

    def _parse_single_sentence(x):
        a = x[0]
        b = x[1]
        a_len = tf.minimum(tf.size(a), seq_length - 3 - tf.size(b))
        input_word_ids = tf.concat(
            [[cls_id], a[:a_len], [sep_id], b, [sep_id]], axis=0
        )
        input_mask = tf.ones_like(input_word_ids, dtype=tf.int64)
        input_type_ids = tf.ragged.row_splits_to_segment_ids(
            [0, a_len + 2, tf.size(input_word_ids)]
        )

        return input_word_ids, input_mask, input_type_ids

    input_word_ids, input_mask, input_type_ids = tf.map_fn(
        _parse_single_sentence,
        [segment_a, segment_b],
        fn_output_signature=(
            tf.RaggedTensorSpec([None], tf.int64),
            tf.RaggedTensorSpec([None], tf.int64),
            tf.RaggedTensorSpec([None], tf.int64),
        ),
    )

    return {
        "input_word_ids": input_word_ids.to_tensor(shape=[batch_size, seq_length]),
        "input_mask": input_mask.to_tensor(shape=[batch_size, seq_length]),
        "input_type_ids": input_type_ids.to_tensor(shape=[batch_size, seq_length]),
    }

to_export.__call__ = call
to_export.call_2 = tf.Module()
to_export.call_2.__call__ = call_2
to_export.call_2.tokenizer = tokenizer
to_export.call_2.sep_id = sep_id
to_export.call_2.cls_id = cls_id

tf.saved_model.save(to_export, 'kcbert-base/preprocess/0')
