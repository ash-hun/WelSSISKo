<div align="center">
  <img src="./assets/logo02.png" width="40%" />
  <h2>WelSSiSKo</h2>
  <p> Welfare Domain Specific Model</p>
</div>

---
## 🔥 Inference Link
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ash-hun/WelSSISKo/blob/main/WelSSiSKo_Inference.ipynb)

## ✅ Model Links

👉 [**WelSSiSKo_v3_llama-2-ko-base_text-generation (Feb.11 2024)**](https://huggingface.co/Ash-Hun/WelSSiSKo_v3_llama-2-ko-base_text-generation)  
👉 [**WelSSiSKo-Chat (Nov.30 2023)**](https://huggingface.co/Ash-Hun/WelSSiSKo-Chat)  
👉 [**WelSSiSKo (Nov.11 2023)**](https://huggingface.co/Ash-Hun/WelSSiSKo)

## 🐶 WelSSiSKo
- 24년 2월 11일 기준으로 `beomi/llama-2-ko-7b`를 베이스로 하여 4bit LoRA를 적용한 Welfare Domain specific model을 작성하였습니다. Instruction Finetuning을 진행하였고 자세한 파일은 huggingface에 올려두었으니 참고하시길 바랍니다.


### 📋 Data
- 학습에 사용된 데이터는 [Welfare QA](https://huggingface.co/datasets/Ash-Hun/Welfare-QA)를 가공하여 사용하였습니다.


## 📆 Updates

<details>
  <summary><strong>Feb.11, 2024</strong></summary>

  - `beomi/llama-2-ko-7b` 기반으로 4bit LoRA를 이용한 Chat Model로 재학습
</details>

<details>
  <summary><strong>Nov.30, 2023</strong></summary>

  - `beomi/polyglot-ko-12.8b-safetensors` 기반으로 8bit LoRA를 이용한 Chat Model로 재학습
</details>
<details>
  <summary><strong>Nov.21, 2023</strong></summary>

  - First Complete Version Upload
</details>


## 🔗 Citation
    @misc{welssisko,
          author = {ash-hun and Noveled},
          title = {WelSSiSKo : Welfare Domain Specific Korean Language Model},
          year = {2023},
          publisher = {GitHub},
          journal = {GitHub repository},
          howpublished = {\url{https://github.com/ash-hun/WelSSISKo}},
        }

## 👥 Contributors

<a href="https://github.com/ash-hun/WelSSISKo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ash-hun/WelSSISKo" />
</a>


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
