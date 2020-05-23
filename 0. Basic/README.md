# Tensorflow에서 RNN관련 API의 사용법에 대해 알아보자.

## Introduction
- 먼저 rnn cell과 rnn의 차이점을 살펴보자.
- Tensorflow에서 rnn cell이 있고, 이 rnn cell을 연결하여 layer로 만든 것이 rnn이다. cell을 묶어 놓음으로써 time step을 batch로 처리 가능한다.
![decode](./rnncell.png)
![decode](./RNN.png)
- RNN의 각 time step의 입력 data는 [batch_size, input_dim]=[N,D]형태이지만, 모든 time_step(즉 seq_length만큼)을 모으면, [batch_size, seq_length, input_dim]=[N,T,D] shape을 가진다. 
- batch로 묶인 data의 sequence길이가 다른 경우에는 padding을 통해, 같은 길이로 맞추는 과정이 필요하다.
- embedding전의 data가 [batch_size, seq_length]형태를 가지는데, embedding을 통해, [batch_size, seq_length, input_dim]형태로 변환된다.
```
import tensorflow as tf

batch_size = 3
seq_length = 5
input_dim = 7
hidden_dim = 4

rnn_cell = tf.keras.layers.LSTMCell(hidden_dim)  # RNN Cell
rnn = tf.keras.layers.RNN(rnn_cell,return_sequences=True) # RNN

initial_state =  rnn.get_initial_state(inputs)  # RNN의 initial state를 0으로 만든다. [batch_size, hidden_dim]

inputs = tf.random.normal([batch_size, seq_length, input_dim])  # Embedding을 거친 data라 가정.
output = rnn(inputs,initial_state)

```
- inital_state는 주로 0으로 만들기도 하고, 다른 정보가 있으면 넣을 수도 있다. inital_state의 shape은 [batch_size, hidden_dim].
- `return_sequences=True`로 하면, [batch_size, seq_length, hidden_dim] shape의 output이 return된다..
- `return_sequences=False`로 했을 때는 sequence의 마지막 값이 [batch_size, hidden_dim] shape의 output이 return된다.

======================================================================================================================================================================


## LSTM & GRU
- Vanilla RNN은 hidden state에 그 전 모든 time step의 정보를 압축하여 다음 step으로 전달하는 구조이다.
- LSTMM, GRU는 직전 time step의 정보와 더 장기적인(더 이전 time step) 정보를 현재 time step에 전달할 수 있도록 모델 구조가 설계되어 있다. 
- Vanilla RNN이 hidden state만으로 구성된 반면, LSTM은 hidden state(short term memory)와 장기 기억 전달을 목적으로 하는 cell state(long term memory) 2개로 구성되어 있다.

======================================================================================================================================================================


## Multi-Layer RNN
- xx


======================================================================================================================================================================

### Bidirectional RNN
- xx

