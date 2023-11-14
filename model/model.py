# 모듈 임포트
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.core import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, GPT2LMHeadModel

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
parser = argparse.ArgumentParser(description='WelfareChatbot based on KoGPT-2')

trainer_config = {
    'max_len': 1024,
    'batch_size': 96,
    'lr': 5e-5,
    'warmup_ratio': 0.1,
    'max_epochs': 3,  # 예시로 3을 설정했습니다
    'gpus': 1,  # GPU 사용 개수 설정 (예시로 1을 설정했습니다)
    'chat':True,
    'sentiment':'0',
    'model_params': 'model_chp/model_-last.ckpt',
    'train':False
}

logger = logging.getLogger()
logger.setLevel(logging.INFO)

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

# 토크나이저 파일들이 저장된 디렉토리 경로
tokenizer_directory = "./tokenizer/"

# 저장된 토크나이저를 다시 로드
domain_tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory, bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token=MASK)

# 토크나이저를 사용하여 텍스트를 토큰화하는 예시
text = "국민취업지원제도를 신청하고 싶은데 어떻게 해야하죠?"
tokenized_text = domain_tokenizer.tokenize(text)
print(tokenized_text)

class CharDataset(Dataset):
    def __init__(self, chats, max_len):
        self._data = chats
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.max_len = max_len
        self.tokenizer = domain_tokenizer 

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn['Q']
        a = turn['A']
        sentiment = str(turn['label'])
        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token + sentiment)   
        q_len = len(q_toked)
        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.mask,
        ] * q_len + a_toked[1:]
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        self.max_len
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
        return(token_ids, np.array(mask),labels_ids)
    
class KoGPT2Chat(LightningModule):
    trainer_config = {
        'max_len': 1024,
        'batch_size': 96,
        'lr': 5e-5,
        'warmup_ratio': 0.1,
        'max_epochs': 3,  # 예시로 3을 설정했습니다
        'gpus': 1,  # GPU 사용 개수 설정 (예시로 1을 설정했습니다)
        'chat':True,
        'sentiment':'0',
        'model_params': 'model_chp/model_-last.ckpt',
        'train':False
    }
    def __init__(self, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.save_hyperparameters(kwargs)
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=32,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=96,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=trainer_config['lr'], correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * trainer_config['max_epochs']
        num_warmup_steps = int(num_train_steps * trainer_config['warmup_ratio'])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = pd.read_csv('./data/ourdata.csv')
        self.train_set = CharDataset(data, max_len=trainer_config['max_len']) # max_len 바꿔주기.
        train_dataloader = DataLoader(
            self.train_set, batch_size=trainer_config['batch_size'], num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)# batch_size 바꿔주기.
        return train_dataloader

    def chat(self, sent='0'):
        tok = domain_tokenizer
        sent_tokens = tok.tokenize(sent)
        with torch.no_grad():
            p = input('user > ')
            q = p.strip()
            a = ''
            while 1:
                input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                pred = self(input_ids)
                gen = tok.convert_ids_to_tokens(
                torch.argmax(
                    pred,
                    dim=-1).squeeze().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace('▁', ' ')
            print("Chatbot > {}".format(a.strip()))
        return q

checkpoint_callback = ModelCheckpoint(
    dirpath='model_chp_last',
    filename='model_{epoch:02d}-{train_loss:.2f}',
    verbose=True,
    save_last=True,
    monitor='train_loss',
    mode='min',
)
print('='*50)
print("Create Model Instace")
model = KoGPT2Chat(hparams=trainer_config)
model.to(device).train()
print('='*50)
print()
print('='*50)
print("Show Model")
print(model)
print('='*50)

# Trainer 클래스 인스턴스 생성
trainer = Trainer(
    max_epochs=trainer_config['max_epochs'],
    devices="auto",
    gradient_clip_val=1.0,
    callbacks=checkpoint_callback,
)

# 학습 시작
trainer.fit(model.to(device))
