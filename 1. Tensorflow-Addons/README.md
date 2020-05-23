# Tensorflow Addons, Encoder-Decoder & Attention

## Tensorflow Addons
```
pip install tensorflow-addons
```

```
import tensorflow as tf
import tensorflow_addons as tfa
```

- Addons에는 Tensorlow의 핵심 API에서 사용할 수 없는 기능을 추가적으로 구현되여 모여있다. Tensorflow 1.x 버전에서는 tensorflow.contrib가 이런 역할을 했다.
- Addons API에서 RNN관련 API로 `tfa.seq2seq`가 있다. 
	* tfa.seq2seq.BasicDecoder
	* tfa.seq2seq.TrainingSampler, tfa.GreedyEmbeddingSampler
	* tfa.seq2seq.dynamic_decode
- tensorflow 1.x의 contrib에 같은 기능을 하는 API가 있다.
	* tf.contrib.seq2seq.BasicDecoder
	* tf.contrib.seq2seq.TrainingHelper,   tf.contrib.seq2seq.GreedyEmbeddingHelper
	* tf.contrib.seq2seq.dynamic_decode
![decode](./BasicDecoder.png)

```
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


vocab_size = 6  # [SOS_token, 1, 2, 3, 4, EOS_token]
SOS_token = 0
EOS_token = 5

x_data = np.array([[SOS_token, 3, 1, 4, 3, 2],[SOS_token, 3, 4, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
y_data = np.array([[3, 1, 4, 3, 2,EOS_token],[3, 4, 2, 3, 1,EOS_token],[1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
print("data shape: ", x_data.shape)


output_dim = vocab_size
batch_size = len(x_data)
hidden_dim =7

seq_length = x_data.shape[1]
embedding_dim = 8

embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,trainable=True) 

inputs = embedding(x_data)
target = tf.convert_to_tensor(y_data)

# Decoder
decoder_cell = tf.keras.layers.LSTMCell(hidden_dim)

#init_state = [tf.zeros((batch_size,hidden_dim)), tf.ones((batch_size,hidden_dim))]   # (h,c)
init_state = decoder_cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32)
    
projection_layer = tf.keras.layers.Dense(output_dim)

sampler = tfa.seq2seq.sampler.TrainingSampler()  # alias ---> sampler = tfa.seq2seq.TrainingSampler()

decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)
outputs, last_state, last_sequence_lengths = decoder(inputs,initial_state=init_state, sequence_length=[seq_length]*batch_size,training=True)


logits = outputs.rnn_output  #(batch_size, seq_length, vocab_size)

print(logits.shape)


weights = tf.ones(shape=[batch_size,seq_length])
loss = tfa.seq2seq.sequence_loss(logits,target,weights)  # logit: (batch_size, seq_length, vocab_size),        target, weights: (batch_size, seq_length)

print('loss: ', loss)
```
