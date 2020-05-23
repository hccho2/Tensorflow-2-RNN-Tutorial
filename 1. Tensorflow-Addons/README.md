# RNN with Tensorflow 2

## Tensorflow Addons
- 먼저 tensorflow에서는 rnn cell(e.g. tf.keras.layers.SimpleRNN)과 rnn(e.g. tf.keras.layers.RNN)을 구분하고 있다.
- rnn cell이 있고, 이 rnn cell을 연결하여 layer로 만든 것이 rnn이다. cell을 묶어 놓음으로써 time step을 batch로 처리 가능한다.