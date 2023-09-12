from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
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
device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True, verbose=False)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
train_dataset = DialogData(os.path.join("/content/data", f'ijcnlp_dailydialog_cc/train/dialogues_train.txt'),
                               tokenizer, neg_per_positive=10)
train_dataset = TRLWrapper(train_dataset)
# print(train_dataset[100])

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
# print("CUSTOM DATASET", custom_dataset[100])

class CustomTransformer(nn.Module):
    def __init__(self, vocab_size=100257, max_seq_length=1024, num_layers=8, num_heads=8, embed_dim=1024, ff_dim=4096, dropout=0.1):
        super(CustomTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.positional_encoding(max_seq_length, embed_dim)

        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attention_mask):
        input_embedded = self.embedding(input_ids)
        input_embedded = input_embedded.to(input_ids.device)
        attention_mask = attention_mask.to(device)
        # tgt_mask = self.generate_square_subsequent_mask(input_ids.size(0)).to(input_ids.device)
        for layer in self.decoder_layers:
            input_embedded = layer(input_embedded, input_embedded, tgt_mask=attention_mask, memory_mask=attention_mask)
        output = self.fc(input_embedded)
        return output

    def positional_encoding(self, max_seq_length, embed_dim):
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class AutoregressiveTransformer(nn.Module):
    def __init__(self, vocab_size=100257, max_seq_length=1024, num_layers=8, num_heads=8, embed_dim=1024, ff_dim=4096, dropout=0.1):
        super(AutoregressiveTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.positional_encoding(max_seq_length, embed_dim)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        input_embedded = self.embedding(input_ids)
        input_embedded = input_embedded.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        
        tgt_mask = self.generate_square_subsequent_mask(input_ids.size(1)).to(input_ids.device)
        
        output = self.decoder(input_embedded, memory=input_embedded, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        output = self.fc(output)  # [batch_size, seq_len, vocab_size]
        
        return output

    def positional_encoding(self, max_seq_length, embed_dim):
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

## SEQ2SEQ Transformer Model

from model import LitS2S

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

# Load the configuration for the base model
# config = AutoConfig.from_pretrained("t5-base")

# Instantiate your custom Transformer model
custom_model = CustomTransformer()
custom_model.to(device)

model = AutoregressiveTransformer()
model.to(device)
## TRAIN PARAMS

train_params = {
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 0
        }

training_loader = DataLoader(custom_dataset, **train_params)
optimizer = torch.optim.AdamW(params =  custom_model.parameters(), lr=1e-6)

## LOSSES

def mle_loss(model, batch, mask):
    print(batch)
    longer_sample = torch.cat((batch['input_ids'].cuda(),batch['target_ids'].cuda()), 1)
    # mask = batch['attention_mask'].cuda()+batch['target_attention_mask'].cuda()
    print("LONGER SAMPLE: ", longer_sample)
    inp = longer_sample[:, :-1]
    mask = torch.ones((1,1))
    print("INPUT: ", inp.shape)
    print("MASK: ", mask.shape)
    model_output = model(inp, mask)
    target = longer_sample[:, 1:]
    print("TARGET: ", type(target))
    logits = model_output[0]
    lprobs = F.log_softmax(logits, dim=-1)
    # lprobs = torch.argmax(lprobs, axis=1)
    # target = target.view(2047, -1)
    print('LPROBS: ', lprobs.shape)
    # assert lprobs.size(0) == 1, 'We work on flat sequences'
    loss = F.nll_loss(lprobs, target[0], reduction='mean')

    '''
    target - [1, 2047]
    target[0] - [2047]
    lprobs - [2047, vs]

    '''

    # true_token_logits = -F.nll_loss(logits[0], target[0], reduction='none')
    ntokens = inp.numel()

    # logging_output = TrainingMetrics.ranking_metrics(logits[0], true_token_logits, None, ntokens, target[0])
    # logging_output['loss'] = loss.item()
    # logging_output['normalizer'] = ntokens
    # logging_output['sample_size'] = ntokens
    # logging_output['ntokens'] = ntokens

    loss = loss / ntokens
    return loss

## Helper Functions

def batch_input_sequence_by_prefix_length(input_sequence, prefix_length):
    seq_len = input_sequence.size(1)
    # Discard tokens if the sequence length is not divisible by the prefix length.
    new_seq_len = (seq_len//prefix_length)*prefix_length
    input_sequence = input_sequence[:, :new_seq_len]
    batch = input_sequence.view(-1, prefix_length).contiguous()
    return batch

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.size(0) == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    logits = logits.squeeze(0)
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def ngram_repeat_mask(xs, n):
    mask = torch.zeros_like(xs)
    for i, x in enumerate(xs):
        seen = set()
        xl = x.tolist()
        for j in range(len(x)-n):
            ng = tuple(xl[j:j+n])
            if ng in seen:
                mask[i, j:j+n] = 1
            seen.add(ng)
    return mask

def sample_sequence(model, prefix_batch, prefix_length, continuation_length, top_k, top_p):
    continuation_logits = []
    context = prefix_batch
    assert context.size(1) == prefix_length
    prev = context
    output = context
    # past = None
    for i in range(continuation_length):
        if(prev.size(1)==50): 
         logits = model(prev[:, :-1], trg=prev[:, 1:])
        elif(prev.size(1)==1): logits = model(prev, max_decode_len=2)
        # logits = model(prev)
        # print("LOGIT SHAPE: ", logits.shape)
        logits = logits[:, -1, :]
        if top_k == 1 and top_p == 0:
            prev = logits.argmax(dim=1, keepdim=True)
            # print("P SHAPE: ", prev.shape)
        else:
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            prev = F.softmax(filtered_logits, dim=-1).multinomial(num_samples=1)

        continuation_logits.append(logits)
        output = torch.cat((output, prev), dim=1)

    continuation_logits = torch.stack(continuation_logits, 1)
    return output, continuation_logits

## ULL Loss

def ul_seq(model, batch, mask):
    input_sequence = torch.cat((batch['input_ids'].cuda(),batch['target_ids'].cuda()), 1)
    print("INPUT SEQ: ", tokenizer.decode(input_sequence[0]))
    batch = batch_input_sequence_by_prefix_length(input_sequence, 50)
    # print("BATCH SEQ: ", batch.shape)
    completions, continuation_logits = sample_sequence(model, batch,
                                                       50, 51, 1, 0.0)
    pred_toks = completions[:, 50:].contiguous()
    # print("PT Shape: ", pred_toks.shape)
    print("PRED TOKS: ", tokenizer.decode(pred_toks.view(-1)))
    mask = ngram_repeat_mask(pred_toks, 4).type_as(continuation_logits)

    lprobs = F.log_softmax(continuation_logits, dim=-1)
    pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
    one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=1e-20).view(pred_toks.size(0), pred_toks.size(1))
    loss = -torch.log(one_minus_probs) * mask
    loss = loss.sum()
    ntokens = pred_toks.numel()  # number of output tokens (tokens in completions)

    # logging_output = {
    #     'seq_loss': loss.item(),
    #     'seq_sample_size': ntokens,
    #     'seq_ntokens': ntokens,
    #     'seq_nsentences': batch.size(0),
    #     'seq_repeat_mask': mask.sum().item(),
    # }

    # # Sum each statistic, which will be normalized by the number of sentences in `aggregate_logging_outputs`.
    # stats = defaultdict(float)
    # for tok_list in pred_toks.cpu().tolist():
    #     ms = ngram_metrics(tok_list)
    #     for k, v in ms.items():
    #         stats[k] += v
    # for k, v in stats.items():
    #     logging_output[k] = v

    loss = loss / ntokens
    return loss
    
    
## unlikelihood loss at token level

def ul_token_loss(model, batch, iteration):
  input = batch['input_ids'].cuda()
  target = batch['target_ids'].cuda()
  output = model(input, trg=target)
  
  if iteration%100==0:
    print("INPUT: ", tokenizer.decode(input[0]))
    print("TARGET: ", tokenizer.decode(target[0]))
    print("OUTPUT: ",  tokenizer.decode(output[0].argmax(dim=1, keepdim=True).view(-1)))

  lprobs = F.log_softmax(output, dim=-1)
  # non_padding_indices = (target != tokenizer.pad_token_id).nonzero()
  # ntokens = len(target[non_padding_indices].squeeze(1))
  ntokens = target.numel()
  # print("LPROBS: ", lprobs.shape)
  loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=tokenizer.pad_token_id)
  flattened_output = output.view(-1, output.size(-1))
  flattened_target = target.view(-1)

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


## TRAINER

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.bool)
        tgt_mask = data['target_attention_mask'].to(device, dtype = torch.bool)

        mask = torch.cat((mask,tgt_mask), 1)
        # outputs = model(input_ids = ids, attention_mask=mask, target_ids=y)

        ## MLE Loss
        
        # loss = mle_loss(model, data, mask)
        
        ## UL Loss
        
        loss = ul_token_loss(model, data, _)
        
        if _%100==0:
          print(f'Loss:  {loss}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

## TRAIN LOOP
print("TRAINING STARTED")
for epoch in range(2):
    train(epoch, tokenizer, seq_model, device, training_loader, optimizer)








