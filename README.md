# **Hotel Reviews Sentiment Analysis**

![Python](https://img.shields.io/badge/Python-3.9-%23007ACC?style=flat-square&logo=python&logoColor=white)  
![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-%23EE4C2C?style=flat-square&logo=pytorch&logoColor=white)  
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-%2334D058?style=flat-square&logo=Hugging%20Face&logoColor=white)  
![License](https://img.shields.io/badge/License-MIT-%23FF9900?style=flat-square)

## **📜 프로젝트 개요**

이 프로젝트는 **호텔 리뷰 데이터셋**을 활용하여, **리뷰 텍스트**를 기반으로 **호텔 지점별로 긍정적인 리뷰와 부정적인 리뷰**를 예측하고, 이를 통해 **각 지점의 평균 평점**과 **예상 평점**을 분석하는 것입니다.  
MobileBERT 모델을 사용하여 **텍스트 분류** 작업을 진행하고, **예측된 평점과 실제 평점 간의 상관관계**를 분석합니다.

---

## **🧑‍💻 기술 스택**

- **Python**: 주요 프로그래밍 언어
- **PyTorch**: 딥러닝 모델 학습 및 예측
- **Transformers**: Hugging Face의 MobileBERT 모델
- **Pandas**: 데이터 처리
- **NumPy**: 수치 계산
- **Matplotlib & Seaborn**: 데이터 시각화

---

## 2. 원시 데이터


[호텔 리뷰들 데이터셋](https://www.kaggle.com/datasets/arnabchaki/tripadvisor-reviews-2023)<br/>


---

## **📊 데이터셋**

- **파일명**: `cleaned_sampled_12_reviews_final.csv`
- **주요 컬럼**:
  - `Text`: 리뷰 텍스트
  - `Branch`: 호텔 지점
  - `Rating`: 실제 평점 (1~5)
  - `Label`: 긍정(1) / 부정(0) 라벨 (평점 > 3이면 긍정, 이하 부정)

---

## **⚙️ 실행 방법**

### **1. 라이브러리 설치**
필요한 라이브러리를 설치하려면 다음 명령어를 실행하세요:

```bash
pip install torch transformers pandas numpy matplotlib seaborn tqdm
```
---

##  2. 코드 설명
사전 세팅
Device 설정: GPU를 사용할 수 있으면 CUDA를 이용하고, 그렇지 않으면 CPU를 사용합니다.

```bash
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
---

## 3. 데이터 로드
CSV 파일을 로드하고, 결측치를 제거한 후, Rating을 실수형으로 변환하고, 긍정/부정 라벨링을 진행합니다.
```bash
df = pd.read_csv("cleaned_sampled_12_reviews_final.csv")
df = df.dropna(subset=["Text", "Branch", "Rating"])
df["Label"] = df["Rating"].apply(lambda x: 1 if x > 3 else 0)
```
---


## 4. 모델 로드
MobileBERT 모델을 로드하고, 해당 모델을 사용하여 리뷰 텍스트의 긍정/부정을 예측합니다.
```bash
model = MobileBertForSequenceClassification.from_pretrained("mobilebert_hotel_finetuned")
tokenizer = MobileBertTokenizer.from_pretrained("mobilebert_hotel_finetuned")
model.to(device)
model.eval()
```
---


## 5. 리뷰 예측
리뷰 텍스트를 토크나이징하여 모델에 입력하고, 각 리뷰에 대해 긍정/부정 예측을 수행합니다.
```bash
inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
```
---

## 6. 지점별 평점 계산
각 지점별로 실제 평점의 평균을 계산하고, 예측된 긍정 비율을 기반으로 예상 평점을 계산합니다.
```bash
actual_ratings = grouped["Rating"].mean()
positive_ratio = grouped["Predicted"].mean()
estimated_ratings = positive_ratio * 4 + 1
```
---


## 7.상관계수 계산
실제 평점과 예측 평점 간의 상관관계를 계산하여 신뢰도를 분석합니다.
```bash
correlation = result["Actual_Avg_Rating"].corr(result["Estimated_Rating"])
```
---


## 8. 시각화
실제 평점과 예상 평점 간의 관계를 시각화하여 직관적으로 비교할 수 있습니다.
```bash
sns.scatterplot(x="Actual_Avg_Rating", y="Estimated_Rating", data=result, hue=result.index)
```
---


## 9. 📈 분석 결과
상관계수 분석
상관계수: 실제 평점과 예측 평점 간의 상관관계를 계산하여, 예측 모델의 신뢰도를 분석합니다.

상관계수 > 0.75: 신뢰도가 높음

상관계수 0.4 ~ 0.75: 중간 정도의 신뢰도

상관계수 < 0.4: 신뢰도가 낮음

##10. 시각화 결과
지점별 실제 평점과 예상 평점의 관계를 시각화하여, 모델 예측이 실제 평점과 어느 정도 일치하는지 확인할 수 있습니다.

## 11. 🚀 개선 방안
파인튜닝: MobileBERT 모델을 현재 데이터셋에 맞게 추가로 파인튜닝할 수 있습니다.

클래스 불균형 처리: 긍정과 부정 리뷰의 비율 불균형 문제를 해결하기 위해 샘플링 기법을 적용하거나, 클래스 가중치를 조정할 수 있습니다.

다중 클래스 분류: 평점이 1부터 5까지의 정수 값을 갖는 경우, 다중 클래스 분류로 모델을 확장할 수 있습니다.

---
## 12. 🚀 결론과 유출 가능한 추론


---
## 🔗 참고 문서
Hugging Face Transformers Documentation

PyTorch Documentation

### 추가된 배지 설명:

1. **Python 버전 배지**: Python 3.9 버전을 나타내는 배지
2. **PyTorch 배지**: PyTorch 1.9.0 버전 배지
3. **Hugging Face 배지**: Hugging Face의 Transformers 라이브러리 배지
4. **라이센스 배지**: MIT 라이센스 배지





