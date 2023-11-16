# Module Import
import json
from pathlib import Path

import torch
from transformers import ElectraTokenizerFast, ElectraForQuestionAnswering, Trainer, TrainingArguments

from model import KorquadLongQADataset

# ====================================================================================
# Define Config
DATA_DIR = '../data/welssisko_data'
TRAIN_PATH = f'{DATA_DIR}/train.json'
DEV_PATH = f'{DATA_DIR}/test.json'

# model_checkpoint = "monologg/koelectra-base-v3-discriminator"
model_checkpoint = "monologg/koelectra-base-v3-finetuned-korquad"

# Device setting
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ====================================================================================
# Define Function
def read_korquad_v2(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        context = group['context']

        for qa in group['qas']:
            question = qa['question']
            answer = qa['answers']

            contexts.append(context)
            questions.append(question)
            answers.append(answer)

    return contexts, questions, answers

def add_end_idx(answers, contexts):
    for item in answers:
        for answer, context in zip(item, contexts):
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            # sometimes squad answers are off by a character or two – fix this
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx-1:end_idx-1] == gold_text:
                answer['answer_start'] = start_idx - 1
                # When the gold label is off by one character
                answer['answer_end'] = end_idx - 1
            elif context[start_idx-2:end_idx-2] == gold_text:
                answer['answer_start'] = start_idx - 2
                # When the gold label is off by two characters
                answer['answer_end'] = end_idx - 2

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(
            i, answers[i][0]['answer_start']))
        end_positions.append(encodings.char_to_token(
            i, answers[i][0]['answer_end'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({
        'start_positions': start_positions,
        'end_positions': end_positions,
    })

def chat(question, context, model, tokenizer):
    question = "서울의 수도는 어디인가요?"
    context = "서울은 대한민국의 수도입니다."

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
    # Dataset Load
    train_contexts, train_questions, train_answers = read_korquad_v2(TRAIN_PATH)
    val_contexts, val_questions, val_answers = read_korquad_v2(DEV_PATH)

    # Add to Index
    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)

    # Tokenizer Load
    print(f'Start load tokenizer : {model_checkpoint}')
    # tokenizer = ElectraTokenizerFast.from_pretrained(model_checkpoint)
    # 토크나이저 파일들이 저장된 디렉토리 경로
    tokenizer_directory = "./tokenizer/"

    # 저장된 토크나이저를 다시 로드
    tokenizer = ElectraTokenizerFast.from_pretrained(tokenizer_directory)
    print(f'Finish load tokenizer : {model_checkpoint}')

    # Set Train/Validation Encodings
    train_encodings = tokenizer(
        train_contexts,
        train_questions,
        truncation=True,
        padding=True,
    )

    val_encodings = tokenizer(
        val_contexts,
        val_questions,
        truncation=True,
        padding=True,
    )

    # Add Token Positions
    add_token_positions(train_encodings, train_answers)
    add_token_positions(val_encodings, val_answers)

    # Build Datasets
    print(f'Start build datasets')
    train_dataset = KorquadLongQADataset(train_encodings)
    val_dataset = KorquadLongQADataset(val_encodings)
    print(f'Finish build datasets')

    # Set Training config
    training_args = TrainingArguments(
        output_dir='./koelectra-v3-long-qa',
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    # PLM Load
    print(f'Start load model : {model_checkpoint}')
    model = ElectraForQuestionAnswering.from_pretrained(model_checkpoint)
    print(f'Finish load model : {model_checkpoint}')

    # Set Trainer
    trainer = Trainer(
        model=model.to(device),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Trainer Run!
    trainer.train()

    # Set Evaluate Mode
    trainer.evaluate()
    # Save Model
    model.save_pretrained("./my_koelectra_model")