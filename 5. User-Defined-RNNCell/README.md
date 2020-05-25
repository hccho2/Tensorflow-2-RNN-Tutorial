# User Defined RNNCell

## User Defined Layer
- Tensorflow에서 제공되는 `tf.keras.layers.Dense`와 같은 표준화된 layer외에 새로운 layer를 만들거나, 기존에 있는 layer들을 조합한 layer를 만들 수도 있다.
- 새로운 Layer를 만들기 위해서는 `tf.keras.layers.Layer` 또는 `tf.keras.Model`을 상속하여 만들 수 있다. 두가지는 조금 차이가 있지만, 둘 다 가능한 경우가 많다.
	* `tf.keras.layers.Layer`: 새로운 weight를 추가로 만들 필요가 있는 경우
	
		```
		class MyLayer(tf.keras.layers.Layer):
			def __init__(self,...):
				super(MyLayer, self).__init__()   # 이게 있어야 한다.
				...
			
			def build(self, input_shape):
				#self.kernel = self.add_variable("kernel",shape=[int(input_shape[-1]),self.num_outputs])
				...
				
			def call(self, inputs,training=None):
				...
		```
	* `tf.keras.Model`: 기존의 layer들의 조합으로 새로운 layer를 만드는 경우
	
		```
		class MyLayer(tf.keras.Model): 
			# tf.keras.Model은 새로운 weight 없이, 기존의 layer들의 조합으로 새로운 layer를 만들 때 사용하면 좋다.
			def __init__(self,...):
				super(MyLayer, self).__init__()   # 이게 있어야 한다.
				...
			
				
			def call(self, inputs,training=None):
				...
		```
- 다음 코드는 `tfa.seq2seq.BasicDecoder(..., output_layer=projection_layer)`의 output_layer로 User Defined Layer를 만들어서 넘겨보자. 
- 보통 Fully Connected Layer 1개를 RNNCell 다음에 붙히는데, Fully Connected Layer 2개를 연결한 새로운 Layer를 만들어서 붙혀보자.
```
#projection_layer = tf.keras.layers.Dense(decoder_output_dim)
projection_layer = MyProjection2(decoder_output_dim)
```
- 다음은 전체 코드이다.
```
class MyProjection(tf.keras.Model):
    # tf.keras.Model은 새로운 weight 없이, 기존의 layer들의 조합으로 새로운 layer를 만들 때 사용하면 좋다.
    def __init__(self,output_dim):
        super(MyProjection, self).__init__()   # 이게 있어야 한다.
        self.output_dim = output_dim
        self.L1 = tf.keras.layers.Dense(self.output_dim, activation = tf.nn.relu)
        self.L2 = tf.keras.layers.Dense(self.output_dim) 
    
        
    def call(self, inputs,training=None):
        y = self.L1(inputs)
        z = self.L2(y)
        return z


batch_size = 3
encoder_length = 5
encoder_input_dim = 7
hidden_dim = 4

encoder_cell = tf.keras.layers.LSTMCell(hidden_dim)  # RNN Cell
encoder = tf.keras.layers.RNN(encoder_cell,return_sequences=False) # RNN

encoder_inputs = tf.random.normal([batch_size, encoder_length, encoder_input_dim])  # Embedding을 거친 data라 가정.

encoder_outputs = encoder(encoder_inputs) # encoder의 init_state을 명시적으로 전달하지 않으면, zero값이 들어간다.  ===> (batch_size, hidden_dim)
     
decoder_length = 10
decoder_input_dim = 11
decoder_output_dim = 8

decoder_cell = tf.keras.layers.LSTMCell(hidden_dim)  # RNN Cell

projection_layer = MyProjection(decoder_output_dim)

sampler = tfa.seq2seq.sampler.TrainingSampler()
decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)

decoder_inputs = tf.random.normal([batch_size, decoder_length, decoder_input_dim])  # Embedding을 거친 data라 가정.
initial_state =  [encoder_outputs,encoder_outputs]  # (h,c)모두에 encoder_outputs을 넣었다.

decoder_outputs = decoder(decoder_inputs, initial_state=initial_state,sequence_length=[decoder_length]*batch_size,training=True)
print(decoder_outputs)
```
## RNNCell 제작하기
- Custom RNNCell은 `tf.keras.layers.Layer`를 상속받아 만드는 것이 좋다. 
- `tf.keras.Model`을 상속받아 만드는 경우 custom cell을 `tf.keras.layers.RNN`에 넘겼을 때, Error가 발생한다.
- __init__() 내에, `self.state_size`, `self.output_size`가 정의되어 있어야 한다.
- 다음 Residual 구조를 가지는 custom RNNCell을 만든 후, Test한 코드이다.
```
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class MyCell(tf.keras.layers.Layer):
    # Residual cell
    def __init__(self, hidden_dim):
        super(MyCell, self).__init__(name='')
        self.hidden_dim = hidden_dim
        self.rnn_cell = tf.keras.layers.LSTMCell(hidden_dim)
        
        self.state_size = self.rnn_cell.state_size
        self.output_size = hidden_dim  # self.rnn_cell.output_size

    def call(self, inputs, states,training=None):
        output, states = self.rnn_cell(inputs,states)
        output = output + inputs
        return output,states

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        
        return [tf.zeros((batch_size,self.hidden_dim),dtype=dtype), tf.zeros((batch_size,self.hidden_dim),dtype=dtype)]

# User Defined cell인 MyCell test
batch_size = 3
seq_length = 4
feature_dim = 7
hidden_dim = feature_dim


cell = MyCell(hidden_dim)   # User Defined Cell

print('-'*20, 'Test 1','-'*20)
# 1 time step 처리
inputs = tf.random.normal([batch_size, feature_dim])
states =  cell.get_initial_state(inputs=None, batch_size=batch_size,dtype=tf.float32)
outputs, states = cell(inputs,states,training=True)
print(outputs)
print(states)


print('-'*20, 'Test 2','-'*20)
# 여러 step을 loop로 처리
inputs = tf.random.normal([batch_size, seq_length, feature_dim])

states =  [tf.zeros([batch_size,hidden_dim]),tf.zeros([batch_size,hidden_dim])]
outputs_all = []
for i in range(seq_length):
    outputs, states = cell(inputs[:,i,:], states)
    outputs_all.append(outputs)


outputs_all = tf.stack(outputs_all,axis=1)
print(outputs_all)
print(states)


print('-'*20, 'Test 3','-'*20)
# tf.keras.layers.RNN을 만들어 batch로 처리.
rnn = tf.keras.layers.RNN(cell,return_sequences=True, return_state=True)

inputs = tf.random.normal([batch_size, seq_length, feature_dim])
states = rnn.get_initial_state(inputs)

whole_seq_output, final_memory_state, final_carry_state = rnn(inputs,states)

print(whole_seq_output.shape, whole_seq_output)
print(final_memory_state.shape, final_memory_state)
print(final_carry_state.shape, final_carry_state)
```
