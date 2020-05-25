# BeamSearchDecoder & Sampler

## Beam Search란
-  decoder의 각 step에서 출력값은 softmax를 취한 확률이고, 이 확률 중에서 가장 높은 값을 선택(argmax)하여 다음 step의 input으로 feeding한다. 이러한 방식을 Greedy Search라고 하다.
- Greedy Search를 좀 더 확장한 방식이 Beam Search라도 할 수 있는데, Beam Search는 search beam 또는 beam width(e.g 3)을 먼저 정한 후, search beam만큼의 복수개를 선택하여 다음 단계로 넘어간다.
- 번역 모델같은 것에서는 정답이 1개만 있는 것이 아니기 때문에 선택의 가능성을 넓혀줄 수 있다.
![decode](./beam-search.png)


