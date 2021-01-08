import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub

raw_bert = hub.load('./kcbert-base/model/0')
raw_preprocess = hub.load('./kcbert-base/preprocess/0')

bert = hub.KerasLayer(raw_bert, trainable=True)
preprocess = hub.KerasLayer(raw_preprocess, arguments={"seq_length": 48})

input_node = tf.keras.Input([], dtype=tf.string)
preprocessed = preprocess(input_node)
output_node = bert(preprocessed)
model = tf.keras.Model(input_node, output_node)
model(tf.constant(['ㅋㅋㅋㅋㅋ 재밌다']))
tf.saved_model.save(model, './tmp/tmp-1')

print("=====================")
print("=====================")
print("=====================")
raw_preprocess = hub.load('./kcbert-base/preprocess/0')
preprocess = hub.KerasLayer(raw_preprocess.call_2, arguments={"seq_length": 48})
input_node = [tf.keras.Input([], dtype=tf.string), tf.keras.Input([], dtype=tf.string)]
preprocessed = preprocess(input_node)
output_node = bert(preprocessed)
model = tf.keras.Model(input_node, output_node)
model(
    [
        tf.constant(['ㅋㅋㅋㅋㅋ 재밌다']),
        tf.constant(['뭐가??', '어떻게 되긴, 개발자 되는 거지 뭐'])
    ]
)
tf.saved_model.save(model, './tmp/tmp-2')
