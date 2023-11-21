# Module Import
import json
import argparse, sys
from pathlib import Path

import torch, gc
from transformers import ElectraTokenizerFast, ElectraForQuestionAnswering, Trainer, TrainingArguments

from model import WelSSiSKo_QADataset

# ====================================================================================
# Default Config
BASE_DIR = '../data/welssisko_data'
TRAIN_PATH = f'{BASE_DIR}/train.json'
DEV_PATH = f'{BASE_DIR}/test.json'
TOKENIZER_PATH = "./tokenizer/"

model_checkpoint = "monologg/koelectra-base-v3-finetuned-korquad"

## Device setting
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

## CLI option setting
parser = argparse.ArgumentParser()
parser.add_argument('-option', help=' : train or test', default='train')
args = parser.parse_args()

# ====================================================================================
# Define Function
def read_data(path):
    path = Path(path)
    with open(path, 'rb') as f:
        korquad_form_dict = json.load(f)

    contexts, questions, answers = [], [], []
    
    for policy in korquad_form_dict['data']:
        context = policy['context']

        for qa in policy['qas']:
            question = qa['question']
            answer = qa['answers']

            contexts.append(context)
            questions.append(question)
            answers.append(answer)

    return contexts, questions, answers

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        exact_answer = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(exact_answer)

        # 문맥에서 정답을 찾아내는 오차범위 감안하여 맵핑
        if context[start_idx:end_idx] == exact_answer:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == exact_answer:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1
        elif context[start_idx-2:end_idx-2] == exact_answer:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2

def add_token_positions(encodings, answers):
    start_pos, end_pos = [], []

    for i in range(len(answers)):
        start_pos.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_pos.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        
        # 인덱스 위치가 제대로 부여되지 않을 경우 최대길이로 처리
        if start_pos[-1] is None:
            start_pos[-1] = tokenizer.model_max_length
        
        if end_pos[-1] is None:
            end_pos[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_pos, 'end_positions': end_pos})

def chat(question, context, model, tokenizer):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    print(answer)

# Main
if __name__ == "__main__":
    # GPU Memeory Empty
    gc.collect()
    torch.cuda.empty_cache()

    # Args Parsing
    argv = sys.argv

    if args.option == 'train': # train 옵션
        # ================================================================================================
        # Ⅰ. Dataset Load & Preprocessing
        # Loading KorQuad form Domain Dataset
        train_contexts, train_questions, train_answers = read_data(TRAIN_PATH)
        val_contexts, val_questions, val_answers = read_data(DEV_PATH)

        # Dataset Log
        print("="*50)
        print(f"Activate Device : {device}")
        print("Sample KorQuad form Domain Dataset. Check below.")
        print(train_answers[:5])
        print("="*50)

        # 리스트를 한 겹 벗기기
        train_answers = [item[0] for item in train_answers]
        val_answers = [item[0] for item in val_answers]

        # Adding End Index
        add_end_idx(train_answers, train_contexts)
        add_end_idx(val_answers, val_contexts)

        # 음수 값이 있는지 확인
        has_negative_values = any(item['answer_start'] < 0 or item['answer_end'] < 0 for item in train_answers)
        print("음수 값이 있는가?", has_negative_values)

        # 음수 값이 있는지 확인
        has_negative_values = any(item['answer_start'] < 0 or item['answer_end'] < 0 for item in val_answers)
        print("음수 값이 있는가?", has_negative_values)

        # ================================================================================================
        # Ⅱ. Domain Specific Tokenizer Load & Encoding
        # Tokenizer Load
        print(f'Start load tokenizer : {model_checkpoint}')

        # 저장된 토크나이저를 다시 로드
        tokenizer = ElectraTokenizerFast.from_pretrained(TOKENIZER_PATH, model_max_length=512)
        print(f'Finish load tokenizer : {model_checkpoint}')
        print("="*50)
        print(" ▼ Tokenizing Test ▼ ")
        print(tokenizer.tokenize("국민취업지원제도를 신청하고 싶은데 어떻게 해야하죠?"))
        print("="*50)

        train_encodings = tokenizer(
            train_contexts,
            train_questions,
            truncation=True,
            padding=True,
            max_length=512
        )

        val_encodings = tokenizer(
            val_contexts,
            val_questions,
            truncation=True,
            padding=True,
            max_length=512
        )

        # ================================================================================================
        # Ⅲ. After Encoding, Token Position Setting
        # Adding Token Positions
        add_token_positions(train_encodings, train_answers)
        add_token_positions(val_encodings, val_answers)

        # ================================================================================================
        # Ⅳ. Datasets Transform inherite 'torch.utils.data.Dataset'
        # Building Datasets
        print(f'| Start build QA Datasets Form... |')
        train_dataset = WelSSiSKo_QADataset(train_encodings)
        val_dataset = WelSSiSKo_QADataset(val_encodings)
        print(f'| Finish build QA Datasets Form! |')

        # ================================================================================================
        # Ⅴ. Ready to train
        # Setting Train Config
        training_args = TrainingArguments(
            output_dir='./koelectra-v3-long-qa',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )

        # Loading Backbone PLM
        print(f'Start load model : {model_checkpoint}')
        model = ElectraForQuestionAnswering.from_pretrained(model_checkpoint)
        print(f'Finish load model : {model_checkpoint}')

        # Setting Trainer Object
        trainer = Trainer(
            model=model.to(device),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # ================================================================================================
        # Run!
        trainer.train()

        # ================================================================================================
        # Set Evaluate Mode
        trainer.evaluate()

        # Save Model
        model.save_pretrained("./my_model")

    elif args.option == 'test':# test 옵션
        print('test')