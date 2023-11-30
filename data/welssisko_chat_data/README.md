# **Ⅰ. Data Explanation**

- **`DataSet`**
    - 공통 형태 : `instruction` - `input` - `output`
        - `instruction` : 실제 질문
        - `input` : 답변에 참조할 문맥
        - `output` : 생성될 답변
        
    - 데이터 예시 ▼
        ```
            [
                {
                    "instruction": "65세 이상 어르신들이 폐렴구균 예방접종을 받으려면 어디로 가야 하나요?",
                    "input": "",
                    "output": "전국 지정의료기관 또는 보건소를 방문하시면 됩니다. 신분증을 지참해야 하며, 예방접종도우미 누리집에서 지정의료기관을 확인할 수 있습니다."
                },
                {
                    "instruction": "혹시 주민등록상 1958년 12월 31일 이전 출생자만 국가예방접종 지원 사업의 대상인가요?",
                    "input": "",
                    "output": "네, 그렇습니다. 65세 이상 어르신 중 주민등록상 출생연도 기준으로 1958년 12월 31일 이전 출생자가 대상입니다."
                },
                {
                    "instruction": "인플루엔자 4가 백신을 매년 몇 회 지원받을 수 있나요?",
                    "input": "",
                    "output": "인플루엔자 4가 백신은 매년 1회 지원받을 수 있습니다."
                },
                {
                    "instruction": "폐렴구균 23가 다당 백신은 몇 회 접종을 지원하나요?",
                    "input": "",
                    "output": "폐렴구균 23가 다당 백신은 1회만 지원됩니다."
                },
                {
                    "instruction": "예방접종을 받기 전에 미리 접종 가능 여부를 확인해야 하나요?",
                    "input": "",
                    "output": "네, 접종기관 방문 전에 접종 가능 여부를 확인하는 것이 필요합니다."
                },
                {
                    "instruction": "질병관리청 콜센터나 예방접종관리과에 문의할 수 있는 번호를 알려주세요.",
                    "input": "",
                    "output": "질병관리청 콜센터는 ☎1339로, 예방접종관리과는 ☎043-719-8398~8399로 문의하실 수 있습니다."
                }
            ]
        ```

---

### **Ⅱ. How to generate?**
    작성일 : 2023.11.30.(THR)
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

                위 내용을 반영한 한국어 질문-응답-문맥 으로 이루어진 데이터셋 샘플 10개를 생성해줘.
                내용은 절대 중복되면 안돼. 질문앞에는 (Q)가 붙고, 응답 앞에는 (A)가 붙어. 
                만들어진 응답 뒤에 무조건 아래 내용을 뒤에 이어붙여줘. 제도명은 위 내용의 제목을 사용하면 돼.

                '이와 관련한 제도는 <<제도명>>입니다. 참고해주세요.'
                마지막으로 문맥에는 위 내용의 제목을 사용하면 돼.
                ```
    - HF(=Human Feedback) 적용
        - 오탈자 및 특수기호 제거
            - 수집된 데이터가 pdf기반이었고, 이를 `.md`파일로의 변환을 마친 뒤 한번 더 정제하고자 함.
            - 해당과정에서 필연적으로 markdown 문법이 포함되는 현상 발생.
            - 즉, markdown 문법을 주력으로 한 자주 사용되지 않는 특수기호에 대해서 제거함.
        - LLM이 약식으로 표현한 내용 보완
        - 항목화 된 내용을 문단으로 정리
    - 기존 한국어 질문답변 오픈소스 데이터셋과 결합하여 총 `11438개` 데이터 활용