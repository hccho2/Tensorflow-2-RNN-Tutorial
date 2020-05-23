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



