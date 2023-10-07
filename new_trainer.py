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
import wandb
from collections import defaultdict, Counter
device = 'cuda' if cuda.is_available() else 'cpu'

wandb.init(project="BTP_ULL_Training")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True, verbose=False)
train_dataset = DialogData(os.path.join("./data", 'ijcnlp_dailydialog_cc/train/dialogues_train.txt'),
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
        reddit_steps_per_epoch=500000,
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
optimizer = torch.optim.Adam(params=seq_model.parameters(), lr=1e-4)

####################################################
def div(x, y):
    if y == 0:
        return x
    else:
        return x / y
        
class NGramIterator:
    """
    N-Gram iterator for a list.
    """

    def __init__(self, lst, n):
        self.lst = lst
        self.n = n
        self.max = len(lst) - n

    def __iter__(self):
        self.counter = -1
        return self

    def __next__(self):
        self.counter += 1
        if self.counter > self.max:
            raise StopIteration
        return tuple(self.lst[self.counter : self.counter + self.n])
        
def _count_n_grams(token_lst, n):
	n_grams = defaultdict(int)
	for n_gram in NGramIterator(token_lst, n):
	    n_grams[n_gram] += 1
	return n_grams

def compute_loss(model, batch, iteration):
	input = batch['input_ids'].cuda()
  	target = batch['target_ids'].cuda()
  	
  	'''
	if (torch.rand(1).item() >= 0.5):
	    total_loss, model_output = super().compute_loss(batch, return_output=True) # Return MLE Loss
	    # No sequence level unlikelihood
	    if return_output:
		return total_loss, model_output
	    else:
		return total_loss
	'''
	# Generate
	clamp_min = 1e-6 # at fp16 since using gpu
	maxlen = 64
	
	################################################
	'''
	with torch.no_grad():
	    beam_pred_scores, _ = self._generate(batch, self.beam_size, maxlen) # put model generation here (LitS2S here)

	# forward pass to create graph for beam search case
	generations = [g[1:] for (g, s, _) in beam_pred_scores] # get the predicted sequences for each input sequence in the minibatch
	pred_toks = torch.nn.utils.rnn.pad_sequence(generations, batch_first=True) # pad the generated sequences to get the true labels ( consider them as true labels now)
	model_output = self.model(*self._model_input(batch), ys=pred_toks) # pass the true labels and the model input to the model to decode and generate outputs
	logits, preds, _ = model_output # get the logits for the model decoing token wise
	'''
	################################################
	
	
	beam_pred_scores = model(input, max_decode_len=maxlen, epsilon=1)['sequence']
	print("SHAPE OF PREDICTED BEAMS: ", beam_pred_scores.shape) # should be [bs, max_len]
	
	generations = [g[1:] for g in beam_pred_scores] # should be [bs, max_len-1]
	pred_toks = torch.nn.utils.rnn.pad_sequence(generations, batch_first=True)
	model_output = model(input, trg=pred_toks)
	logits = model_outputs['logits']
	print("SHAPE OF LOGITS: ", logits.shape)
	
	if iteration%200==0:
        print("INPUT: ", tokenizer.decode(input[0]))
        print("TARGET: ", tokenizer.decode(target[0]))
        print("OUTPUT: ",  tokenizer.decode(logits[0].argmax(dim=1, keepdim=True).view(-1)))	
	

	# construct mask marking repeats
	n = 4  # label n-grams
	crep_mask = torch.zeros_like(pred_toks).type_as(logits) # create mask for context repititions
	lrep_mask = torch.zeros_like(pred_toks).type_as(logits) # create mask for responce repititions

	for i, gen in enumerate(generations):
	    gen_i = gen.tolist() # generation was a single dimensional tensor

	    # Collect context ngrams
	    # batch - dictionary {input, output}
	    context_i = batch.input_ids[i].tolist()
	    context_n_grams = _count_n_grams(context_i, n)

	    seen_n_grams = defaultdict(int)

	    # penalize if there is a context repeat
	    for j, n_gram in enumerate(NGramIterator(gen_i, n)):
		if context_n_grams[n_gram] > 0:
		    crep_mask[i, j : j + n] = 1

	    # penalize if there is a label repeat
	    for j, n_gram in enumerate(NGramIterator(gen_i, n)):
		if seen_n_grams[n_gram] > 0:
		    lrep_mask[i, j : j + n] = 1
		seen_n_grams[n_gram] += 1

	# Compute unlikelihood loss - we can keep this part entirely same ig
	pred_logsoftmax = torch.nn.LogSoftmax(dim=2)
	lprobs = pred_logsoftmax(logits)
	pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
	one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=clamp_min).view(
	    pred_toks.size(0), pred_toks.size(1)
	)

	mask = (0.5 * lrep_mask) + (
	    0.5 * crep_mask
	)

	ul_loss = -(torch.log(one_minus_probs)) * mask
	total_loss = div(ul_loss.sum(), mask.sum())
	#self.record_local_metric(
	#    'ul_loss', AverageMetric.many(ul_loss.sum(dim=-1), mask.sum(dim=-1))
	#)

	#if not self.is_training:
	#    # in eval mode, we want metrics (e.g. PPL) provided by tga's compute_loss
	#    _, _ = super().compute_loss(batch, return_output=True)

	# if return_output:
	#    return total_loss, model_output
	return total_loss
####################################################

def ul_token_loss(model, batch, iteration):
  input = batch['input_ids'].cuda()
  target = batch['target_ids'].cuda()
  # output = model(input, trg=target)
  op = model(input, max_decode_len=64, epsilon=-1)
  output = op['logits']

  # print(output.shape)
  
  if iteration%500==0:
    print("INPUT: ", tokenizer.decode(input[0]))
    print("TARGET: ", tokenizer.decode(target[0]))
    print("OUTPUT: ",  tokenizer.decode(output[0].argmax(dim=1, keepdim=True).view(-1)))
    # print("OUTPUT: ", tokenizer.decode(output[0]))

  # lprobs = F.log_softmax(output, dim=-1)
  # non_padding_indices = (target != tokenizer.pad_token_id).nonzero()
  # ntokens = len(target[non_padding_indices].squeeze(1))
  ntokens = target.numel()
  # print("LPROBS: ", lprobs.shape)
  loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=tokenizer.pad_token_id)
  l = output.shape[1]
  sliced_target = target[:, :64]
  # print(l, output.shape, target.shape, sliced_target.shape)
  flattened_output = output[:, :, :].reshape(output.shape[0]*(l), output.shape[2]) 
  flattened_target = sliced_target[:, 1:].reshape(sliced_target.shape[0]*(l))

  mle_loss = loss_fn(flattened_output, flattened_target)
  
  # tar = target.view(-1)
  # # print("TAR: ", tar.shape)
  # ctx_cands = tar.unsqueeze(0).expand(tar.size(0), tar.size(0))
  # ctx_cands_ = (ctx_cands.tril(-1) + tokenizer.pad_token_id)
  # ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
  # ctx_cands = ctx_cands.tril(-1) + ctx_cands_

  # ctx_cands = ctx_cands.masked_fill(ctx_cands == tar.unsqueeze(1), tokenizer.pad_token_id)
  # negative_targets = torch.zeros_like(lprobs.view(-1, lprobs.size(-1))).scatter_(1, ctx_cands, 1)

  # one_minus_probs = torch.clamp((1.0 - lprobs.view(-1, lprobs.size(-1)).exp()), min=1e-5)
  # ul_loss = -torch.log(one_minus_probs)*negative_targets
  # ul_loss = ul_loss.sum()

  # loss = mle_loss + ul_loss

  wandb.log({
      "MLE LOSS: ": mle_loss,
      # "ULL LOSS: ": ul_loss,
      # "TOTAL LOSS:": loss
      })
  # loss = loss/ntokens

  return mle_loss

def train(epoch, tokenizer, model, device, loader, optimizer):
  model.train()
  for _,data in enumerate(loader, 0):
      ## MLE Loss
      
      # loss = mle_loss(model, data, mask)
      
      ## UL Loss
      
      loss = compute_loss(model, data, _)
      
      if _%200==0:
          print('Loss: ', loss)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

## TRAIN LOOP
print("TRAINING STARTED")
for epoch in range(20):
    print("EPOCH: ", epoch)
    train(epoch, tokenizer, seq_model, device, training_loader, optimizer)

wandb.finish()
