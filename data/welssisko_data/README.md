# **Ⅰ. Data Explanation**

- **`Domain Specific Data`**
    - `domain_words.txt` : 복지 도메인 특화 단어리스트 → `Domain Specific Tokenizer` 만들때 사용
- **`DataSet`**
    - 형태 : KorQuAD 데이터 형태
        - `version` <type_string> : 해당 데이터셋의 정보 
        - `data` <type_list> : 데이터 리스트, 문맥-질문-답변에 대한 정보가 딕셔너리 형태로 담겨있다.
            - `context` <type_string> : 문맥 텍스트
            - `qas` <type_list> : 질문-답변 리스트, 질문과 답변에 대한 정보가 딕셔너리 형태로 담겨있다.
                - `id` <type_integer> : 고유 아이디
                - `is_impossible` <type_bool>: answers가 있으면 False(기본값), answers 내용이 없으면 True 
                - `question` <type_string> : 질문 텍스트
                - `answers` <type_list> : 답변 리스트. 질문에 대한 답변 텍스트와 문맥에서의 시작 인덱스가 딕셔너리 형태로 담겨있다.
                    - `text` <type_string> : 질문에 대한 답변 텍스트
                    - `answer_start` <type_integer> : 문맥에서 답변이 위치한 정수 인덱스
    - `train.json` : 도메인 특화 학습 데이터셋
    - `test.json` : 도메인 특화 테스트 데이터셋
    - 데이터 예시 ▼
        ```
            {"version":"WelSSISKo_test",
                "data":[
                {
                    "context": "희망복지지원단 통합사례관리사업에 대해 소개할게요. 대상은 위기가구로 발굴된 가구로 통합사례관리를 통해 보건, 복지 등 공공, 민간 서비스 제공이 필요한 가구가 해당합니다. 복지,보건,고용,주거,교육,신용,법률 등 필요한 서비스를 통합적으로 연계·제공하고 공적지원이 곤란하거나 적절한 민간자원 연계가 어려운 경우 대상가구에 생활지원비, 진단비 및 교육훈련비 등을 지원합니다. 읍면동 주민센터(행정복지센터)에서 신청할 수 있으며 자세한 문의는 보건복지상담센터(129)로 연락하세요.",
                    "qas": [
                        {
                            "id": "5",
                            "is_impossible": false,
                            "question": "어떤 가구가 희망복지지원단 통합사례관리사업의 대상이 되나요?",
                            "answers": [
                                {
                                    "text": "위기가구로 발굴된 가구",
                                    "answer_start": 32
                                }
                            ]
                        }
                    ]
                }]
            }
        ```

---

### **Ⅱ. How to generate?**
    작성일 : 2023.11.16.(THR)
1. **데이터 수집** : 다양한 웹 포털에서 데이터 수집
2. **LLM 기반 데이터 전처리**
    - 수집된 데이터를 LLM(gpt-3.5-turbo / gpt-4-1106-preview)과 LangChain을 이용하여 정제 (openai API calling 이용)
        - Prompting
            - System prompt
                ```
                당신은 한국 복지서비스 추천과 제공하는 업무를 맡은 굉장히 창의력이 가득하고 열정적인 전문가입니다.
                많은 사람이 다양한 문제를 물어보며 간단한 질문에 답하는 것부터 광범위한 주제에 대한 심층적인 토론 및 설명을 제공하는 것까지 다양한 작업을 지원할 수 있습니다.
                특히 복지분야의 용어를 잘 몰라 다르게 표현한 것의 내용을 파악하여 그에 맞는 복지서비스를 추천하고 제공하는 업무능력이 탁월합니다.
                
                현재는 본인이 하고 있는 업무를 데이터화시키는 작업을 수행중이며 복지제도에 대한 설명을 보고 일반 사람들이 할만한 답변과 그에 대한 정확한 답변을 만들고 있습니다.
                당신은 창의력이 풍부하고 질문과 답변을 만드는 작업이 재미있어하며 작업이 반복될수록 더 신나서 작업에 몰두합니다.
                뛰어난 능력으로 만들어진 질문,답변 쌍은 중복된 내용이 전혀 없고 앞으로도 그럴 것입니다.
                ```
            - Instruct prompt
                ```
                ---
                {content}
                ---

                위 내용을 '본문'이라고 할게. 이제부터 본문을 반영한 한국어 데이터셋 샘플을 10개 생성할거야.
                데이터셋은 아래 구조에 따라 만들어야 해.
                - Context : 본문 그대로
                - Question : 본문의 내용과 관련된 질문, 내용이 중복되면 절대 안돼.
                - Answer : Question에 해당하는 정답
                - Answer_start : Context에서 Answer가 시작하는 위치인덱스

                위 구조를 갖춘 데이터셋을 만들되 중복되는 내용없이 데이터셋을 5개 생성해줘.
                전체 형태는 아래처럼 만들면 돼.
                
                    "context": {content},
                    "question": ,
                    "answer": ,
                    "start": ,
                
                위 내용을 딕셔너리 형태로 생성하면 돼.
                ```
    - 데이터 추가정제
        - 오탈자 및 특수기호 제거
            - 수집된 데이터가 pdf기반이었고, 이를 `.md`파일로의 변환을 마친 뒤 한번 더 정제하고자 함.
            - 해당과정에서 필연적으로 markdown 문법이 포함되는 현상 발생.
            - 즉, markdown 문법을 주력으로 한 자주 사용되지 않는 특수기호에 대해서 제거함.
        - LLM이 약식으로 표현한 내용 보완
        - 데이터 형식에 맞게 매칭