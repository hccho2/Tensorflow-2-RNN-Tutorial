# User Defined RNNCell

## User Defined Layer
- Tensorflow에서 제공되는 `tf.keras.layers.Dense`와 같은 표준화된 layer외에 새로운 layer를 만들거나, 기존에 있는 layer들을 조합한 layer를 만들 수도 있다.
- 새로운 Layer를 만들기 위해서는 `tf.keras.layers.Layer` 또는 `tf.keras.Model`을 상속하여 만들 수 있다. 두가지는 조금 차이가 있지만, 둘 다 가능한 경우가 많다.
	* `tf.keras.layers.Layer`: 새로운 weight를 추가로 만들 필요가 있는 경우
	
		```
		class MyLayer(tf.keras.layers.Layer):  # tf.keras.layers.Layer    tf.keras.Model
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
		class MyLayer(tf.keras.Model):  # tf.keras.layers.Layer    tf.keras.Model
			# tf.keras.Model은 새로운 weight 없이, 기존의 layer들의 조합으로 새로운 layer를 만들 때 사용하면 좋다.
			def __init__(self,...):
				super(MyLayer, self).__init__()   # 이게 있어야 한다.
				...
			
				
			def call(self, inputs,training=None):
				...
		```



## RNNCell 제작하기



