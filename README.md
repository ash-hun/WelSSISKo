<div align="center">
  <img src="./assets/logo02.png" width="40%" />
  <h2>WelSSISKo</h2>
  <p> Welfare Domain Specific Model</p>
</div>

---
## ✅ Download Links

👉 [**더 정확하고 자세한 예시는 huggingface Inference API Widget참조**](https://huggingface.co/Ash-Hun/WelSSiSKo)
```python
from transformers import pipeline

REPO = 'Ash-Hun/WelSSiSKo'

q1 = "BLOOM은 몇 개의 프로그래밍 언어를 지원합니까?"
c1 = "BLOOM은 1,760억 개의 매개변수를 보유하고 있으며 46개 언어 자연어와 13개 프로그래밍 언어로 텍스트를 생성할 수 있습니다."

q2 = "안녕?  내 이름은 Ash야. 너의 취미는 뭐니?"
c2 = "안녕 나를 소개하지 이름 김하온 직업은 travler 취미는 tai chi, meditation, 독서, 영화시청 랩 해 터 털어 너 그리고 날 위해 증오는 빼는 편이야 가사에서 질리는 맛이기에 나는 텅 비어 있고 prolly 셋 정도의 guest 진리를 묻는다면 시간이 필요해 let me guess 아니면 너의 것을 말해줘 내가 배울 수 있게 난 추악함에서 오히려 더 배우는 편이야 man"

question_answer = pipeline("question-answering", model=REPO)
print(question_answer(question=q1, context=c1, truncation=True))
print(question_answer(question=q2, context=c2, truncation=True))
```
```
{'score': 0.965011477470398, 'start': 44, 'end': 47, 'answer': '13개'}
{'score': 0.04200083762407303, 'start': 55, 'end': 65, 'answer': '독서, 영화시청 랩'}
```

## 👥 Contributors

<a href="https://github.com/ash-hun/WelSSISKo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ash-hun/WelSSISKo" />
</a>

## WelSSiSKo Model
- 본 모델은 빅리더 AI인턴쉽에서 주관한 SSiS와의 프로젝트 진행중 '도메인 특화 모델'제작에 대한 가능성 입증을 위해 만든 것입니다. 복지 도메인에 특화된 Chat-Model로 기획하였으나 복합적인 이유로 QA-Model로 방향을 바꾸어 제작되었습니다.
- 추가적으로 `beomi/polyglot-ko-12.8b-safetensors`를 바탕으로 chatmodel로 파인튜닝한 모델을 huggingface에 올려두었으니 참고하시길 바랍니다.

#### Backbone Model
- [KoELECTRA-base-v3-finetuned-korquad](https://huggingface.co/monologg/koelectra-base-v3-finetuned-korquad)
- 프로젝트의 본 목적이 도메인에 특화된 내용을 바탕으로 QA를 수행하고자 하였고, 다수의 한국어 PLM중에서 고민하였습니다.
- 주어진 Compute Power와 기간(약 10일), 난이도, 데이터셋 생성 등 다양한 복합요소를 고려하여 가장 범용적으로 활용되는 모델을 선정하게 되었습니다.

#### Vocabulary
- 복지 도메인에 특화된 단어를 아래 정성적 기준에 맞추어 선정하였습니다.
  - 해당 도메인에서만 사용되는 단어
    `ex) 국민취업지원제도, 급식카드`
  - 일반단어가 해당 도메인에서 더 중요한, 혹은 다른 의미로 사용된 단어
    `ex) 재발급`
  - 해당 도메인에서만 사용되는 신조어
    `ex) 취준, 국비`
- 총 Domain Specific Vocab `467`개

#### Tokenizer
- Backbone Model(=KoELECTRA계열)에 맞게 `WordPiece`를 사용하였습니다.
- Tokenizer Sample
  ```python
  sample = "아이가 급식카드를 분실했어요. 어떻게 재발급 받을 수 있나요?"
  sample2 = "국민취업지원제도를 신청하고 싶은데 어떻게 해야하죠?"
  sample3 = "취준하고있는 학생입니다. 경제적으로 너무 부담스럽네요. 제가 받을 수 있는 지원금은 없나요...?"

  print("[Sample Text 01] : 아이가 급식카드를 분실했어요. 어떻게 재발급 받을 수 있나요?")
  print(f"[Before] : {tokenizer_electra.tokenize(sample)}")
  print(f"[After] : {welssisko.tokenize(sample)}")
  print("="*150)
  print("[Sample Text 02] : 국민취업지원제도를 신청하고 싶은데 어떻게 해야하죠?")
  print(f"[Before] : {tokenizer_electra.tokenize(sample2)}")
  print(f"[After] : {welssisko.tokenize(sample2)}")
  print("="*150)
  print("[Sample Text 03] : 취준하고있는 학생입니다. 경제적으로 너무 부담스럽네요. 제가 받을 수 있는 지원금은 없나요...?")
  print(f"[Before] : {tokenizer_electra.tokenize(sample3)}")
  print(f"[After] : {welssisko.tokenize(sample3)}")
  ```
  ```text
  [Sample Text 01] : 아이가 급식카드를 분실했어요. 어떻게 재발급 받을 수 있나요?
  [Before] : ['아이', '##가', '급식', '##카드', '##를', '분실', '##했', '##어요', '.', '어떻게', '재발', '##급', '받', '##을', '수', '있', '##나', '##요', '?']
  [After] : ['아이', '##가', '급식카드', '를', '분실', '##했', '##어요', '.', '어떻게', '재발', '##급', '받', '##을', '수', '있', '##나', '##요', '?']
  ======================================================================================================================================================
  [Sample Text 02] : 국민취업지원제도를 신청하고 싶은데 어떻게 해야하죠?
  [Before] : ['국민', '##취업', '##지', '##원', '##제', '##도', '##를', '신청', '##하', '##고', '싶', '##은', '##데', '어떻게', '해야', '##하', '##죠', '?']
  [After] : ['국민취업지원제도', '를', '신청', '##하', '##고', '싶', '##은', '##데', '어떻게', '해야', '##하', '##죠', '?']
  ======================================================================================================================================================
  [Sample Text 03] : 취준하고있는 학생입니다. 경제적으로 너무 부담스럽네요. 제가 받을 수 있는 지원금은 없나요...?
  [Before] : ['취', '##준', '##하', '##고', '##있', '##는', '학생', '##입니다', '.', '경제', '##적', '##으로', '너무', '부담', '##스럽', '##네', '##요', '.', '제', '##가', '받', '##을', '수', '있', '##는', '지원금', '##은', '없', '##나', '##요', '.', '.', '.', '?']
  [After] : ['취준', '하고', '##있', '##는', '학생', '##입니다', '.', '경제', '##적', '##으로', '너무', '부담', '##스럽', '##네', '##요', '.', '제', '##가', '받', '##을', '수', '있', '##는', '지원금', '##은', '없', '##나', '##요', '.', '.', '.', '?']

  ```

#### Data
- 데이터 형태는 **KorQuAD**의 형태를 따랐습니다.
- Context에 특정한 복지제도의 내용을 추출하여 **최소한의 전처리**만 하여 넣었습니다.
- Context는 `최소 token 91`, `최대 token 1423`사이로 구성되어있습니다.
- Question과 Answer는 정제된 Context를 이용해 OpenAI의 API (gpt-3.5-turbo / gpt-4-1106-preview)를 활용하여 **자체 생성한뒤 일일이 검수**하였습니다.
- **최종적으로 사용한 `Question-Answer Pair Set은 4134개`입니다.**


## 📆 Updates

<details>
  <summary><strong>Nov.30, 2023</strong></summary>

  - beomi/polyglot-ko-12.8b-safetensors 기반으로 8bit LoRA를 이용한 Chat Model로 재학습
</details>
<details>
  <summary><strong>Nov.21, 2023</strong></summary>

  <!-- summary 아래 한칸 공백 두어야함 -->
  - First Complete Version Upload
</details>

## ✉️ Acknowldgement

- *WelSSiSKo* 는 빅리더 AI인턴쉽과 SSiS가 함께한 프로젝트에서 도메인 특화모델의 가능성을 입증하기위해 만든 모델로써 충분하지 않은 도메인 데이터와 다소 깨끗하지않은 전처리 데이터를 바탕으로 학습된 모델입니다. 이후 천천히 수정되어 나갈 계획이니 양해바랍니다.

## 📑 Reference
- [monologg/KoELECTRA](https://github.com/monologg/KoELECTRA)
- [beomi/KoAlpaca](https://github.com/Beomi/KoAlpaca)
- [beomi/peft](https://github.com/Beomi/peft)
- [GY-Jeong/KoELECTRA-KorQuAD](https://github.com/GY-Jeong/KoELECTRA-KorQuAD)
- [decaf0cokes/KorQuADv2](https://github.com/decaf0cokes/KorQuADv2)
- [sehandev/koelectra-korquad-v2](https://github.com/sehandev/koelectra-korquad-v2)
- [화해 뷰티도메인 PLM](https://blog.hwahae.co.kr/all/tech/tech-tech/5876)
- [Langcon 2023](https://festa.io/events/3097)
- [박장원 : 특정 도메인에 맞는 언어 모델은 어떻게 만들까](https://www.youtube.com/watch?v=N3VDk9pRZuw&ab_channel=Language)
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
