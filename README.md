# 로그


참고 URL : 
- SARSA 코드 예제 : https://jyoondev.tistory.com/154
- Python_RL_Envs : https://github.com/jellyho/Python_RL_Envs/tree/2aab22dc93b3d2e0380080822a11e395db340c41

브랜치 : 


2024.03.15
- feature-snake-torch, feature-torch-colab에서 정상적으로 학습을 한다.
- 학습된 결과는 코랩,로컬(레노버),로컬(맥북) 어디에서나 동작함을 확인할 수 있다.
- torch에서 loss함수에 대한 인자 설정이 잘못되어 DQNTorchAgent도 수정해야한다.
- 이 인자 설정 때문에 제대로 된 학습이 이뤄지지 않았다.
- 코드들은 적당히 깨끗이 정리했다.
- 추가적인 연습은 더 필요없을 것으로 보이며 자료들은 참고 자료로 활용한다.
- 간단한 토치 학습 : dqn_torch_test01
- 토치 기반 뱀 게임 : torch_train


2024.03.12
- feature-snake-torch를 생성하여 토치화한다.
- torch를 기반으로 하기 위해 모델을 정의하고 forward함수를 chatgpt를 기반으로 작성했다.
- 모델에 인자 값들 중 conv를 고려해서 작성해야하는 수식과 개념을 파이토치 노트에 작성했다.
- torch에서 예상하는 인자 순서가 있어, 공부해뒀던 einops를 적용하여 채널의 숫자를 rearrange해줬다.
- tf와 달리 가중치를 저장 및 불러오는 함수가 달라서 반영했다.
- 학습 검증에 시간이 걸리기 때문에 colab화를 병행해서 처리하도록 한다. feature-torch-colab이라는 branch를 생성하여 실험한다.
- 지금 타겟 모델을 검증하고 있기 때문에 유효한 플롯을 만들기 위해 feature-torch-colab을 사용한다. 

2024.03.11
- 실제적인 구현을 위해 current_state의 구조를 노트에 메모하고 분석하게 됐다.
- sarsa를 dqn전에 미리 구현하여 테스트 및 활용한다는 생각은 사실상 어렵다는 것을 알게 됐다.
- sarsa는 dqn을 이해하기 위해 기반이 되는 지식정도로 이해하고 넘어가는 것이 좋겠다.
- sarsa대신 torch를 기반으로 학습을 해보도록 한다.

2024.02.04
- 먼저 sarsa 버전을 먼저 구현하고 있음.
- 조금씩 기능을 완성하는 방향으로 접근함.

2024.02.02
- snake 프로젝트 현재 디텍토리에 추가함.
- SARSA버전과 Torch버전을 구현하고 다음 예제로 넘어가는 것으로 결정.

2024.02.01
- feature-model-save-load 구현 완료
- colab에서 학습한 모델을 로컬에 가져와서 테스트를 마침

2024.01.30
- feature-device-to 구현을 위해 기능을 적용하고 테스트하니 속도가 더 느려지는 현상을 목격했다. 그리고 아래와 같은 결론을 chatgpt를 통해서 받아낼 수 있었다.
```
GPU는 병렬 연산에 강점이 있어, 대규모 데이터셋과 복잡한 모델에서 훨씬 더 빠른 속도를 제공합니다. 
그러나 강화학습에서는 일반적으로 매 스텝마다 모델을 업데이트하고, 이는 상대적으로 작은 배치 크기를 사용합니다. 
이런 경우에는 CPU에서 GPU로의 데이터 전송 오버헤드가 상대적으로 크게 작용하여, GPU를 사용하는 것이 오히려 느려질 수 있습니다.
```
- 환경 시뮬레이션이나 복잡한 모델이 아닐 경우에는 cpu기반으로 돌려도 되겠다.
- feature-device-to 구현 완료
- git diff feature-device-to feature-device-to-comp 를 통해서 device를 반영하는 곳을 비교 분석할 수 있다.


2024.01.29
- feature-epsilon-decaying 구현 완료 

<B>구현 전(좌), 구현 후(우) </B>
<p align="center" width="100%">
<img src="./img/no_epsilon_decaying.png" width="45%"/>
<img src="./img/epsilon_decaying_adapted.png" width="45%"/>

</p>

2024.01.24
- 학습 중 랜덤에 따라 쑥 떨어지는 구간이 생기는 것 같다. 그냥 epsilon이 아니라 epsilon decaying을 접목시켜보자.
- 코랩에서의 모델을 저장하고 로컬에서 반영하는 기능을 테스트하는 것이 필요함.
- 학습 효율성을 위해 무거운 작업을 device에서 할 수 있도록 변경하는 것이 필요함.

2024.01.20
- dqn_torch_test01 : 내가 바로 torch로 직접 하나씩 건드려가며 코드 변환을 함. 하지만 실행자체가 되지 않음.
- dqn_torch_test02 : chatgpt를 이용하여 DQNAgent를 바로 chatgpt로 변환함. 실행은 되지만 학습이 되지 않음.

2024.01.19
- 모델의 input으로 쓰이는 것은 state이다.

2024.01.08
- 취업 및 수익 발생을 위해서는 파이토치가 유리함.
- 파이토치를 학습 후, 토치기반으로 DQN화 하면 됨.

2024.01.04
- 예제로 쓸만한 SARSA를 발견하고 로컬에서 기능을 확인함.
- SARSA를 DQN형태(뱀게임을 참고)로 변환함.
- 로컬에서도 돌아가나 colab에서 빠른 속도를 보여줌.
- 뱀게임은 텐서플로우 기반으로 구성됨.
