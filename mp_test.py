import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

raw_bert = hub.load('./kcbert-base/model/0')
raw_preprocess = hub.load('./kcbert-base/preprocess/0')

bert = hub.KerasLayer(raw_bert, trainable=True, dtype=policy)
preprocess = hub.KerasLayer(raw_preprocess, arguments={"seq_length": 48})

print('Bert dtype: %s' % bert.dtype)

input_node = tf.keras.Input([], dtype=tf.string)
preprocessed = preprocess(input_node)
output_node = bert(preprocessed)
model = tf.keras.Model(input_node, output_node)
result = model(tf.constant(['ㅋㅋㅋㅋㅋ 재밌다']))
print("output dtype: %s" % result['sequence_output'].dtype.name)
tf.saved_model.save(model, './tmp/tmp-mp-1')
