import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel, AutoConfig, AutoTokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from dialog_data import DialogData, TRLWrapper
from torch import cuda
import math
import numpy as np
from model import LitS2S
device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True, verbose=False)
train_dataset = DialogData(os.path.join("/content/data", f'ijcnlp_dailydialog_cc/train/dialogues_train.txt'),
                               tokenizer, neg_per_positive=10)
train_dataset = TRLWrapper(train_dataset)

data = []
for example in train_dataset:
    if(example["query"]==[] or example["response"]==[]):
      continue
    data.append(example)

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        query = example['query']
        response = example['response']

        # Tokenize the query and response using the T5 tokenizer
        inputs = self.tokenizer.encode_plus(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        # Create target inputs (responses) for the decoder
        target_inputs = self.tokenizer.encode_plus(
            response,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_ids = target_inputs['input_ids'].squeeze()
        target_attention_mask = target_inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_ids': target_ids,
            'target_attention_mask': target_attention_mask
        }

custom_dataset = CustomDataset(data, tokenizer)

seq_model = LitS2S(vocab_size=len(tokenizer), 
        pad_idx=tokenizer.pad_token_id,
        bos_idx=tokenizer.cls_token_id,
        eos_idx=tokenizer.sep_token_id,
        num_training_steps=10,
        run_path_root="runs/",
        d_model=256, 
        nhead=8, 
        num_encoder_layers=6, num_decoder_layers=6, 
        dim_feedforward=1024, 
        dropout=0.1,
        max_pos_idx=1024,
        learning_rate=0.0001,
        valuefn_learning_rate=0.001,
        use_lr_scheduler=False,
        sampled_reward_mode=False,
        margin=0.1,
        use_baseline=False,
        vanilla_ce=False,
        equal_reward_alloc=False,
        enc_weight_init=None,
        dec_weight_init=None,
        dataset="dd",
        data_path_root="data/",
        retrieval_model_type="ESIM",
        batch_size=2,
        epochs=10,
        early_stopping_patience=5,
        reddit_steps_per_epoch=500_000,
        device_count=1,
        no_early_stopping=False,
        prob_positive=0.9,
        response_sampler=None,
        vocab="bert",
        max_ctx_len=100,
        max_resp_len=50)
        
seq_model.to(device)

train_params = {
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 0
        }

training_loader = DataLoader(custom_dataset, **train_params)
optimizer = torch.optim.Adam(params =  seq_model.parameters(), lr=1e-4)

def ul_token_loss(model, batch, iteration):
  input = batch['input_ids'].cuda()
  target = batch['target_ids'].cuda()
  output = model(input, trg=target)
  
  if iteration%500==0:
    print("INPUT: ", tokenizer.decode(input[0]))
    print("TARGET: ", tokenizer.decode(target[0]))
    print("OUTPUT: ",  tokenizer.decode(output[0].argmax(dim=1, keepdim=True).view(-1)))

  lprobs = F.log_softmax(output, dim=-1)
  # non_padding_indices = (target != tokenizer.pad_token_id).nonzero()
  # ntokens = len(target[non_padding_indices].squeeze(1))
  ntokens = target.numel()
  # print("LPROBS: ", lprobs.shape)
  loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=tokenizer.pad_token_id)
  l = output.shape[1]
  flattened_output = output[:, :-1, :].reshape(output.shape[0]*(l-1), output.shape[2]) 
  flattened_target = target[:, 1:].reshape(target.shape[0]*(l-1))

  mle_loss = loss_fn(flattened_output, flattened_target)
  
  tar = target.view(-1)
  # print("TAR: ", tar.shape)
  ctx_cands = tar.unsqueeze(0).expand(tar.size(0), tar.size(0))
  ctx_cands_ = (ctx_cands.tril(-1) + tokenizer.pad_token_id)
  ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
  ctx_cands = ctx_cands.tril(-1) + ctx_cands_

  ctx_cands = ctx_cands.masked_fill(ctx_cands == tar.unsqueeze(1), tokenizer.pad_token_id)
  negative_targets = torch.zeros_like(lprobs.view(-1, lprobs.size(-1))).scatter_(1, ctx_cands, 1)

  one_minus_probs = torch.clamp((1.0 - lprobs.view(-1, lprobs.size(-1)).exp()), min=1e-5)
  ul_loss = -torch.log(one_minus_probs)*negative_targets
  ul_loss = ul_loss.sum()

  loss = mle_loss + ul_loss

  # loss = loss/ntokens

  return loss

def train(epoch, tokenizer, model, device, loader, optimizer):
  model.train()
  for _,data in enumerate(loader, 0):
      ## MLE Loss
      
      # loss = mle_loss(model, data, mask)
      
      ## UL Loss
      
      loss = ul_token_loss(model, data, _)
      
      if _%500==0:
        print(f'Loss:  {loss}')
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

## TRAIN LOOP
print("TRAINING STARTED")
for epoch in range(20):
    print("EPOCH: ", epoch)
    train(epoch, tokenizer, seq_model, device, training_loader, optimizer)
