# 딥러닝 & LLM Step by Step 튜토리얼 🚀

딥러닝의 기초부터 대규모 언어 모델(LLM)까지 차근차근 배워보는 한국어 튜토리얼입니다.

## 📚 학습 로드맵

### Step 1: Python과 NumPy 기초
- **파일**: `tutorials/step1_basics/01_python_numpy_basics.ipynb`
- **학습 내용**:
  - Python 기본 자료구조
  - NumPy 배열과 연산
  - 행렬 연산 이해하기
  - 딥러닝에 필요한 수학 함수들

### Step 2: 퍼셉트론 - 신경망의 기초
- **파일**: `tutorials/step2_perceptron/02_perceptron.ipynb`
- **학습 내용**:
  - 퍼셉트론의 구조와 동작 원리
  - 논리 게이트 구현 (AND, OR, XOR)
  - 학습 알고리즘 구현
  - 선형 분류의 한계와 해결법

### Step 3: 다층 퍼셉트론 (MLP) - 딥러닝의 시작
- **파일**: `tutorials/step3_mlp/03_multilayer_perceptron.ipynb`
- **학습 내용**:
  - 순전파(Forward Propagation) 구현
  - 역전파(Backpropagation) 알고리즘
  - 경사하강법으로 학습하기
  - 비선형 문제 해결

### Step 4: PyTorch 기초
- **파일**: `tutorials/step4_pytorch/04_pytorch_basics.ipynb`
- **학습 내용**:
  - PyTorch 텐서와 자동 미분
  - nn.Module로 신경망 구축
  - 데이터로더와 최적화
  - GPU 활용법

### Step 5: CNN - 이미지 인식
- **파일**: `tutorials/step5_cnn/05_cnn_image_classification.ipynb`
- **학습 내용**:
  - Convolution과 Pooling 이해
  - MNIST 손글씨 숫자 분류
  - CIFAR-10 컬러 이미지 분류
  - 전이학습과 데이터 증강

### Step 6: RNN - 순차 데이터 처리
- **파일**: `tutorials/step6_rnn/06_rnn_sequence_processing.ipynb`
- **학습 내용**:
  - RNN, LSTM, GRU 구현
  - 텍스트 생성 모델
  - 감성 분석
  - 시계열 예측

## 🚀 LLM (Large Language Model) 학습 과정

### Step 7: Attention Mechanism - LLM의 핵심
- **파일**: `tutorials/step7_attention/07_attention_mechanism.ipynb`
- **학습 내용**:
  - Attention의 직관적 이해
  - Query, Key, Value 개념
  - Scaled Dot-Product Attention
  - Multi-Head Attention
  - Positional Encoding

### Step 8: Transformer 아키텍처
- **파일**: `tutorials/step8_transformer/08_transformer_architecture.ipynb`
- **학습 내용**:
  - Encoder-Decoder 구조
  - Layer Normalization과 Residual Connection
  - Feed-Forward Networks
  - 완전한 Transformer 구현

### Step 9: Mini GPT 구현
- **파일**: `tutorials/step9_mini_gpt/09_mini_gpt_implementation.ipynb`
- **학습 내용**:
  - GPT 아키텍처 이해
  - 텍스트 토큰화
  - 모델 학습 파이프라인
  - 텍스트 생성 전략

### Step 10: 대규모 LLM 이해와 활용
- **파일**: `tutorials/step10_large_llm/10_understanding_large_llms.ipynb`
- **학습 내용**:
  - 현대 LLM의 발전 과정
  - Hugging Face Transformers 사용법
  - Fine-tuning과 Prompt Engineering
  - LLM의 실제 응용

## 🛠 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/tiny-llm-by-claude.git
cd tiny-llm-by-claude
```

### 2. 가상환경 생성 (권장)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. 필요 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. Jupyter Notebook 실행
```bash
jupyter notebook
```

## 📋 필요 사항

- Python 3.8 이상
- CUDA 지원 GPU (선택사항, 더 빠른 학습을 위해)
- 최소 8GB RAM 권장

## 💡 학습 팁

1. **순서대로 학습하기**: Step 1부터 차근차근 진행하세요.
2. **코드 직접 실행**: 모든 코드를 직접 실행하고 결과를 확인하세요.
3. **연습 문제 풀기**: 각 노트북 끝에 있는 연습 문제를 꼭 풀어보세요.
4. **실험하기**: 파라미터를 바꿔가며 결과가 어떻게 달라지는지 관찰하세요.

## 🎯 학습 목표

이 튜토리얼을 완료하면:
- 딥러닝의 기본 개념을 확실히 이해할 수 있습니다
- NumPy로 신경망을 처음부터 구현할 수 있습니다
- PyTorch를 사용하여 실제 문제를 해결할 수 있습니다
- CNN으로 이미지를 분류하고 RNN으로 텍스트를 처리할 수 있습니다
- Transformer와 Attention 메커니즘을 완벽히 이해할 수 있습니다
- 나만의 Mini GPT를 구현할 수 있습니다
- 대규모 LLM을 활용한 실제 응용 프로그램을 만들 수 있습니다

## 🚦 다음 단계

이 튜토리얼을 마친 후에는:
- **고급 LLM 기법**: LoRA, QLoRA, PEFT
- **멀티모달 AI**: CLIP, DALL-E, Flamingo
- **강화학습**: RLHF, Constitutional AI
- **최신 연구**: Flash Attention, Mixture of Experts
- **실전 프로젝트**: 자신만의 AI 서비스 구축

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

오타 수정, 내용 개선, 새로운 예제 추가 등 모든 기여를 환영합니다!

---

**Happy Learning! 🎉**