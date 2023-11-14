# **Tokenizer Fine Tuning BenchMarking**

1. #### Domain Data를 토큰화해서 Domain Vocab에 저장
2. #### Domain Vocab으로 ELECTRA Tokenizer fine tuning
3. #### Evaluation
    - `Sample Sentence` 설정
    - `ELECTRA Tokenizer (일반)`로 토큰화
    - `ELECTRA Tokenizer (Domain Vocab 포함)`로 토큰화
    - `성능 비교` : Sample Sentence가 Domain Word를 잘 토크나이징 하는지 봅시다!