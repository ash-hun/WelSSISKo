<div align="center">
  <img src="./assets/logo02.png" width="40%" />
  <h2>WelSSiSKo</h2>
  <p> Welfare Domain Specific Model</p>
</div>

---

## ✅ Model Links

👉 [**WelSSiSKo-Chat (Nov.30 2023)**](https://huggingface.co/Ash-Hun/WelSSiSKo-Chat)  
👉 [**WelSSiSKo (Nov.11 2023)**](https://huggingface.co/Ash-Hun/WelSSiSKo)

## 👥 Contributors

<a href="https://github.com/ash-hun/WelSSISKo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ash-hun/WelSSISKo" />
</a>

## 🐶 WelSSiSKo
- 본 모델은 빅리더 AI인턴쉽에서 주관한 SSiS와의 프로젝트 진행중 '도메인 특화 모델'제작에 대한 가능성 입증을 위해 만든 것입니다. 복지 도메인에 특화된 Chat-Model로 기획하였으나 복합적인 이유로 QA-Model로 방향을 바꾸어 제작되었습니다.
- 이후 `beomi/polyglot-ko-12.8b-safetensors`를 바탕으로 Chat-Model로 파인튜닝한 모델을 huggingface에 올려두었으니 참고하시길 바랍니다.

### Backbone Model
- **`WelSSiSKo`**
  - [KoELECTRA-base-v3-finetuned-korquad](https://huggingface.co/monologg/koelectra-base-v3-finetuned-korquad)
  - 프로젝트의 본 목적이 도메인에 특화된 내용을 바탕으로 QA를 수행하고자 하였고, 다수의 한국어 PLM중에서 고민하였습니다.
  - 주어진 Compute Power와 기간(약 10일), 난이도, 데이터셋 생성 등 다양한 복합요소를 고려하여 가장 범용적으로 활용되는 모델을 선정하게 되었습니다.
- **`WelSSiSKo-Chat`**
  - [Polyglot-ko-12.8B-SafeTensors](https://huggingface.co/beomi/polyglot-ko-12.8b-safetensors)
  - LoRA를 활용하여 기존에 사용된 데이터셋과 더불어 새로 생성한 데이터셋을 얹어 사용함

### Vocabulary
- 복지 도메인에 특화된 단어를 아래 정성적 기준에 맞추어 선정하였습니다.
  - 해당 도메인에서만 사용되는 단어
    `ex) 국민취업지원제도, 급식카드`
  - 일반단어가 해당 도메인에서 더 중요한, 혹은 다른 의미로 사용된 단어
    `ex) 재발급`
  - 해당 도메인에서만 사용되는 신조어
    `ex) 취준, 국비`
- 총 Domain Specific Vocab `467`개

### Tokenizer
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

### 📋 Data
- 각 모델에 맞는 데이터셋을 `data/`안에 자세히 기재해놓았으니 참고바랍니다.


## 📆 Updates

<details>
  <summary><strong>Nov.30, 2023</strong></summary>

  - beomi/polyglot-ko-12.8b-safetensors 기반으로 8bit LoRA를 이용한 Chat Model로 재학습
</details>
<details>
  <summary><strong>Nov.21, 2023</strong></summary>

  - First Complete Version Upload
</details>

## ✉️ Acknowldgement

- *WelSSiSKo* 는 빅리더 AI인턴쉽과 SSiS가 함께한 프로젝트에서 도메인 특화모델의 가능성을 입증하기위해 만든 모델로써 충분하지 않은 도메인 데이터와 다소 깨끗하지않은 전처리 데이터를 바탕으로 학습된 모델입니다. 이후 천천히 수정되어 나갈 계획이니 양해바랍니다.

## 🔗 Citation
    @misc{welssisko,
          author = {ash-hun and Noveled },
          title = {WelSSiSKo : Welfare Domain Specific Korean Language Model},
          year = {2023},
          publisher = {GitHub},
          journal = {GitHub repository},
          howpublished = {\url{https://github.com/ash-hun/WelSSISKo}},
        }

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
