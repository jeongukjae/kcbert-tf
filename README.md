# kcbert-tf

KcBERT를 TensorFlow Hub 형태로 가져와서 쓸 수 있게 만든 레포지토리입니다.

코드는 정말 엉망인데,, 모델이 같은지에 대해서는 검증을 거쳤으니 괜찮을 거예요.

TODO

- [ ] mixed_float16 지원 -> 안되네... mp_test.py 참고

## 사용법

```python
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub

bert = hub.KerasLayer("https://github.com/jeongukjae/kcbert-tf/releases/download/base-1/model.tar.gz")
# for single input
preprocess = hub.KerasLayer(
    "https://github.com/jeongukjae/kcbert-tf/releases/download/base-1/preprocess.tar.gz",
    arguments={"seq_length": 128}
)

# 아래처럼 classification 모델 구성가능해요
input_node = tf.keras.Input([], dtype=tf.string)
preprocessed = preprocess(input_node)
bert_output = bert(preprocessed)
output = tf.keras.layers.Dense(2)(bert_output['pooled_output'])
model = tf.keras.Model(input_node, output)

# for multiple input (segment a, b)
# preprocess = hub.load("https://github.com/jeongukjae/kcbert-tf/releases/download/base-1/preprocess.tar.gz")
#
# preprocess = hub.KerasLayer(
#     preprocess.call_2,
#     arguments={"seq_length": 128}
# )
```

## Signature

### <https://github.com/jeongukjae/kcbert-tf/releases/download/base-1/preprocess.tar.gz>

**Single Bert Input**

```text
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_tensor'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_input_tensor:0
    inputs['seq_length'] tensor_info:
        dtype: DT_INT32
        shape: ()
        name: serving_default_seq_length:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['input_mask'] tensor_info:
        dtype: DT_INT64
        shape: (-1, -1)
        name: StatefulPartitionedCall_1:0
    outputs['input_type_ids'] tensor_info:
        dtype: DT_INT64
        shape: (-1, -1)
        name: StatefulPartitionedCall_1:1
    outputs['input_word_ids'] tensor_info:
        dtype: DT_INT64
        shape: (-1, -1)
        name: StatefulPartitionedCall_1:2
```

seq_length를 받는데, 저거는 그냥 hub.KerasLayer 줄때 argument로 주세요

**Multiple Bert Input**

이거는 위랑 똑같은데, `input_tensor`가 `[tf.TensorSpec([None], dtype=tf.string), tf.TensorSpec([None], dtype=tf.string)]` 이런 식이예요.

### <https://github.com/jeongukjae/kcbert-tf/releases/download/base-1/model.tar.gz>

```text
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_mask'] tensor_info:
        dtype: DT_INT64
        shape: (-1, -1)
        name: serving_default_input_mask:0
    inputs['input_type_ids'] tensor_info:
        dtype: DT_INT64
        shape: (-1, -1)
        name: serving_default_input_type_ids:0
    inputs['input_word_ids'] tensor_info:
        dtype: DT_INT64
        shape: (-1, -1)
        name: serving_default_input_word_ids:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['pooled_output'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 768)
        name: StatefulPartitionedCall:0
    outputs['sequence_output'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, -1, 768)
        name: StatefulPartitionedCall:1
```

`input_mask`, `input_type_ids`, `input_word_ids`를 dict로 `[batch_size, sequence length]`형태로 받고, `pooled_output`, `sequence_output`을 dict 형태로 내뱉어요.
