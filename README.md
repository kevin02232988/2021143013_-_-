# Hotel Reviews Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.9-%23007ACC?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-%23EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-%2334D058?style=flat-square&logo=Hugging%20Face&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-%23FF9900?style=flat-square)

## 📜 프로젝트 개요
이 프로젝트는 호텔 리뷰 데이터셋을 활용하여, 리뷰 텍스트를 기반으로 호텔 지점별로 긍정적인 리뷰와 부정적인 리뷰를 예측하고, 이를 통해 각 지점의 평균 평점과 예상 평점을 분석하는 프로젝트입니다. MobileBERT 모델을 사용하여 텍스트 분류 작업을 진행하고, 예측된 평점과 실제 평점 간의 상관관계를 분석합니다.

## 🧑‍💻 기술 스택
- **Python**: 주요 프로그래밍 언어
- **PyTorch**: 딥러닝 모델 학습 및 예측
- **Transformers**: Hugging Face의 MobileBERT 모델
- **Pandas**: 데이터 처리
- **NumPy**: 수치 계산
- **Matplotlib & Seaborn**: 데이터 시각화

## 📊 데이터셋
- **파일명**: `cleaned_sampled_12_reviews_final.csv`
- **주요 컬럼**:
  - `Text`: 리뷰 텍스트
  - `Branch`: 호텔 지점
  - `Rating`: 실제 평점 (1~5)
  - `Label`: 긍정(1) / 부정(0) 라벨 (평점 > 3이면 긍정, 이하 부정)

## ⚙️ 실행 방법

### 1. 라이브러리 설치
필요한 라이브러리를 설치하려면 다음 명령어를 실행하세요:

```bash
pip install torch transformers pandas numpy matplotlib seaborn tqdm







---

### 추가된 배지 설명:

1. **Python 버전 배지**: Python 3.9 버전을 나타내는 배지
2. **PyTorch 배지**: PyTorch 1.9.0 버전 배지
3. **Hugging Face 배지**: Hugging Face의 Transformers 라이브러리 배지
4. **라이센스 배지**: MIT 라이센스 배지

이 배지들은 프로젝트의 주요 기술 스택과 라이센스를 한 눈에 알아볼 수 있게 해 줍니다. `simple-icon-badges`에서 제공하는 다양한 아이콘을 활용하여 프로젝트에 관련된 배지를 자유롭게 추가할 수 있습니다. 

배지를 수정하려면 링크를 자신이 원하는 대로 수정하고 추가하면 됩니다. 예를 들어, **PyTorch 버전**이나 **라이센스** 등은 배지 링크에서 해당 부분만 변경하면 됩니다.

