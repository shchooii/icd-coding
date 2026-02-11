# Mitigating Imbalance in ICD Coding  
### Long-tailed 의료 데이터에서의 손실 함수와 Threshold 최적화 연구

본 저장소는 다음 논문의 공식 구현 코드입니다:

> *Mitigating Imbalance in ICD Coding: A Comprehensive Evaluation of Loss Functions and Threshold Optimization*

## 개요

병원에서는 환자의 진단과 치료 내용을  
**ICD 코드**라는 표준 질병 코드로 기록합니다.

예를 들어:

- 당뇨
- 폐렴
- 심부전
- 희귀 유전 질환

모두 각각의 ICD 코드가 있습니다.

최근에는 이러한 코딩 작업을  
AI로 자동화하려는 연구가 활발합니다.


ICD 코딩은 단순한 분류 문제가 아닙니다.

### 1️⃣ 한 환자에게 여러 코드가 동시에 붙습니다

→ 다중 라벨 분류(multi-label classification)



### 2️⃣ 코드 분포가 매우 불균형합니다

- 흔한 질환 코드: 데이터가 매우 많음  
- 희귀 질환 코드: 데이터가 매우 적음

이를 **long-tailed 분포**라고 합니다.

이 때문에 AI는:

- 흔한 질환만 잘 맞추고  
- 희귀 질환은 잘 예측하지 못하는

문제가 생깁니다.



## 연구 내용

이 연구는:

> “AI가 희귀 질환 코드도 잘 맞추게 하려면  
> 어떤 학습 방법이 좋은가?”

를 체계적으로 분석합니다.

특히 두 가지를 중점적으로 봤습니다:

### ✔ Loss 함수 (학습 방법)

AI가 무엇을 더 중요하게 학습할지 결정하는 요소



### ✔ Threshold 설정

AI의 예측을  
“정답으로 볼지 말지” 정하는 기준값



## 주요 결과

### 1️⃣ 좋은 Loss 함수는 희귀 코드 성능을 올립니다

→ 희귀 질환 예측이 개선됨  
→ 하지만 흔한 질환 성능은 유지됨



### 2️⃣ Threshold 설정이 매우 중요합니다

단순히 0.5를 쓰는 것보다,

코드를 그룹으로 나누어  
각 그룹에 다른 threshold를 적용하면  
더 안정적인 성능을 얻을 수 있습니다.



### 3️⃣ Loss와 Threshold는 함께 써야 합니다

Loss만 바꾸는 것으로는 충분하지 않고,

> Loss + Threshold tuning 조합이 가장 효과적

이라는 것을 보였습니다.



## 사용 데이터

본 연구는 실제 병원 데이터인  
MIMIC 데이터셋을 사용했습니다.

- 수만 명 환자의 퇴원 기록
- 수천~수만 개 ICD 코드
- 극심한 long-tailed 분포

PhysioNet 인증을 통해 접근 가능합니다.



## 사용 방법

### 모델 학습

```bash
python main.py experiment=mimiciv_icd10/plm_icd gpu=0
````

### 평가만 수행

```bash
python main.py \
  experiment=mimiciv_icd10/plm_icd \
  load_model=/path/to/model.ckpt \
  trainer.epochs=0
```
