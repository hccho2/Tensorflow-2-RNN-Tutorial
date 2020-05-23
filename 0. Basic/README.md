# Tensorflow에서 RNN관련 API의 사용법에 대해 알아보자.
- 먼저 rnn cell과 rnn의 차이점을 살펴보자.
- Tensorflow에서 rnn cell이 있고, 이 rnn cell을 연결하여 layer로 만든 것이 rnn이다. 
![decode](./rnncell.png)

- RNN의 input data는 [batch_size, seq_length, input_dim]의 shape을 가진다. 
- batch로 묶인 data의 sequence길이가 다른 경우에는 padding을 통해, 같은 길이로 맞추는 과정이 필요하다.
- embedding전의 data가 [batch_size, seq_length]형태를 가지는데, embedding을 통해, [batch_size, seq_length, input_dim]형태로 변환된다.
```
import tensorflow as tf

batch_size = 3
seq_length = 5
input_dim = 7
hidden_dim = 4
inputs = tf.random.normal([batch_size, seq_length, input_dim])  # Embedding을 거친 data라 가정.
rnn_cell = tf.keras.layers.LSTMCell(hidden_dim)
rnn = tf.keras.layers.RNN(rnn_cell,return_sequences=True)
initial_state =  rnn.get_initial_state(inputs)
output = rnn(inputs,initial_state)

```
- `return_sequences=True`로 하면, [batch_size, seq_length, hidden_dim] shape의 output이 return된다..
- `return_sequences=False`로 했을 때는 sequence의 마지막 값이 [batch_size, hidden_dim] shape의 output이 return된다.
