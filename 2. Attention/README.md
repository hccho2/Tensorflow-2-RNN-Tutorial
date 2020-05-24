# Attention

## Attention이란
- Attention은 Encoder의 마지막 hidden state에 모든 정보를 압축하는 방식의 단점을 보완하기 위한 기법이다.
- Attention은 Encder의 모든 time step에서의 출력을 Decoder에 전달하는 방식이다. 이 Encoder의 모든 출력을 Memory(또는 Key)라 부른다.
![decode](./Attention.png)
	- N: batch_size,
	- Te: Encoder sequence Length
	- eh, dh: Encoder Hidden Dim, Decoder Hidden dim
- Encoder의 모든 hidden state, 즉 Memory는 [N,Te,eh]형태를 가진다. 이 memory는 eh크기의 vector가 Te개 있다고 보면된다. 
- Decoder의 time step i에서의 hidden state를 Query라 부른다. 이 Query는 [N,dh] 형태이다. 
- Te개의 Memory vector 각각과 Query간에 score라는 것을 계산하고 나면, score는 모두 Te개가 된다. 이 score를 계산하는 방식에는 여러 Attention 모델이 있을 수 있다.
	* Dot Product Attention: memory vector와 query를 단순 내적하여 score를 계산한다. 내적을 위해서는 eh=dh가 되어야 한다.
	* Luong Attention: Dot Product Attention을 좀 더 일반화. 
	* Bahdanau Attention
- score에 softmax를 취하면 합이 1이 되는 확률로 볼 수 있는데, 이것을 alignment라 한다. 
- 이 alignment로 Memory vector들을 가중평균(weighted sum)하면 [N,eh] 크기의 vector가 만들어진다. 이 vector를 context라 부른다.
- 다시 context를 가공하여 Attention vector를 만든다. 다시 말해, Attention은 context로 부터 얻어지는데, 가공하지 않고 그대로 사용하기도 한다.
![decode](./score.png)
- 이제 score을 어떻게 계산하는지 자세히 살펴보자.
![decode](./Attention_Score.png)