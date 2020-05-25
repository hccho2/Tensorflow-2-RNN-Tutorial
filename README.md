# Tensorflow 2 RNN-tutorial

<p align="center"><img width="700" src="TF2-RNN.png" />  </p>

- This repository is a tutorial for RNN model developers using Tensorflow.
- 이 tutorial은 tensorflow 1.x 용 [RNN Tutorial](https://github.com/hccho2/Tensorflow-RNN-Tutorial)을 Tensorflow 2에 맞게 수정한 것이다.
- 이 tutorial 코드를 실행하기 위해서는 Tensorflow 2.2.0, tensorflow_addons 0.10.0 필요
- TensorFlow SIG(Special Interest Group) [Addons](https://www.tensorflow.org/addons/overview?hl=ko)은 Tensorflow에서 사용할 수 없는 기능을 추가로 구현한 API를 모아놓은 것이다. 
- tensorflow 1.x 의 tensorflow.contrib.seq2seq는 tensorflow 2.x에서 tensorflow_addons.seq2seq로 변환되었다고 볼 수 있다.
- 여기서는 Addon중에서 RNN관련 API를 살펴볼 예정이다:
    * tfa.seq2seq.Sampler
    * tfa.seq2seq.BasicDecoder, tfa.seq2seq.BeamSearchDecoder
    * tfa.seq2seq.dynamic_decode
    * tfa.seq2seq.BahdanauAttention, tfa.seq2seq.LuongAttention
- Decoder에서 좀 더 다양한 Sampling을 통한 output을 만들어 내기 위해서는 여러가지 Sampler를 다룰 수 있어야 한다.
- 또한, RNNCell, Sampler를 custumization하여 User Defined RNNCell, Sampler를 만들어 보자.


## 0. [Basic RNN Model](https://github.com/hccho2/Tensorflow-2-RNN-Tutorial/tree/master/0.%20Basic)
RNN모델의 기본적인 구조와 이를 위한 Tensorflow API를 살펴본다.
- Introduction & Embedding
- LSTM, GRU
- Multi-Layer RNN
- Bidirectional RNN

### 1. [Tensorflow Addons](https://github.com/hccho2/Tensorflow-2-RNN-Tutorial/tree/master/1.%20Tensorflow-Addons) 
- Tensorflow Addons
- Decoder, Sampler
- Encoder-Decoder(seq2seq) 모델 


### 2. [Attention with Tensorflow](https://github.com/hccho2/Tensorflow-2-RNN-Tutorial/tree/master/2.%20Attention-With-Tensorflow)
- Attention이 개념.
- Dot Product Attention, Luong Attention, Bahdanau Attention
- Tensorflow에서의 Attention 구현 Detail.

### 3.[BeamSearchDecoder & 다양한 Sampler](https://github.com/hccho2/Tensorflow-2-RNN-Tutorial/tree/master/3.%20BeamSearchDecoder-Sampler)
- Beam Search Algorithm과 Tensorflow API `tfa.seq2seq.BeamSearchDecoder`
- Minimal Character Model Train 시키기.
- SampleEmbeddingSampler, ScheduledOutputTrainingSampler

### 4. [User Defined Sampler](https://github.com/hccho2/Tensorflow-2-RNN-Tutorial/tree/master/4.%20User-Defined-Sampler)
- `tfa.seq2seq.InferenceSampler`를 이용하여 Customization해 보자.
- User Defined Sampler: 직접 Sampler를 만들어 보자.

### 5. [User Defined RNNCell](https://github.com/hccho2/Tensorflow-2-RNN-Tutorial/tree/master/5.%20User-Defined-RNNCell)
- User Defined RNNCell: 직접 RNN Cell을 만들어 보자.