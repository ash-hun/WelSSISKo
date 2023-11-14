import torch
from transformers import pipeline, AutoModelForCausalLM

MODEL = 'beomi/KoAlpaca-Polyglot-5.8B'

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=device, non_blocking=True)
model.eval()

pipe = pipeline(
    'text-generation', 
    model=model,
    tokenizer=MODEL,
    device=device
)

def ask(x, context='', is_input_full=False):
    ans = pipe(
        f"### 질문: {x}\n\n### 맥락: {context}\n\n### 답변:" if context else f"### 질문: {x}\n\n### 답변:", 
        do_sample=True, 
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=2,
    )
    print(ans[0]['generated_text'])

ask("딥러닝이 뭐야?")
# 딥러닝은 인공신경망을 통해 입력과 출력 사이의 복잡한 관계를 학습하는 머신러닝의 한 분야입니다. 이 기술은 컴퓨터가 인간의 학습 능력과 유사한 방식으로 패턴을 학습하도록 하며, 인간의 개입 없이도 데이터를 처리할 수 있는 기술입니다. 최근에는 딥러닝을 활용한 인공지능 애플리케이션이 많이 개발되고 있습니다. 예를 들어, 의료 진단 애플리케이션에서는 딥러닝 기술을 활용하여 환자의 특징을 파악하고, 이를 통해 빠르고 정확한 진단을 내리는 데 사용됩니다. 또한, 금융 분야에서는 딥러닝 기술을 활용하여 주가 예측 모형을 학습하는 데 사용되기도 합니다. 