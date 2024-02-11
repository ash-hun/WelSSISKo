import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from peft import PeftModel
from transformers import AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

# Setting Config

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False
)

load_model = AutoModelForCausalLM.from_pretrained("beomi/llama-2-ko-7b", quantization_config=bnb_config, device_map='auto', use_auth_token=HF_TOKEN)
load_model.config.use_cache = False
load_model.config.pretraining_tp = 1

base_tokenizer = AutoTokenizer.from_pretrained("beomi/llama-2-ko-7b", trust_remote_code=True, use_auth_token=HF_TOKEN)
base_tokenizer.pad_token = base_tokenizer.eos_token
base_tokenizer.padding_side = "right"

load_model = PeftModel.from_pretrained(load_model, "Ash-Hun/WelSSiSKo_v3_llama-2-ko-base_text-generation")

def prompt_processing(user_input):
    return f'### Instruction:\n{user_input}\n\n### Response:'

def export_output(ouput_text):
    sep = ouput_text[0]['generated_text'].split('### Response:')[1].split('### Instruction')[0].split('## Instruction')[0].split('# Instruction')[0].split('Instruction')[0]
    sep = sep[1:] if sep[0] == '.' else sep
    sep = sep[:sep.find('.')+1] if '.' in sep else sep
    return sep

gen_pipe = pipeline(task="text-generation",
                model=load_model,
                tokenizer=base_tokenizer,
                max_length=500,
                do_sample=True,
                temperature=0.1,
                num_return_sequences=1,
                eos_token_id=base_tokenizer.eos_token_id,
                top_k=3,
                # top_p=0.3,
                repetition_penalty = 1.3,
                framework='pt'
                # early_stopping=True
)

if __name__ == "__main__":
    prompt = "장애인 복지와 관련된 세제 혜택을 체크할 때 중요한 점은 무엇인가요?"  
    result = gen_pipe(prompt_processing(prompt))  
    print(export_output(result))