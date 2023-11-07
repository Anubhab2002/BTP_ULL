import argparse
from email.policy import default
import json
import os
from collections import defaultdict
from dialog_data import BaseDataClass
import glob
from pprint import pprint

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

from nltk import word_tokenize
from nltk import ngrams
from collections import Counter

# =================== OLD CODE ===================
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import time
import pickle
import matplotlib
#import spacy
from collections import Counter
import os, re, math, copy, time, sys
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm, tqdm_notebook
import subprocess
from matplotlib import pyplot as plt
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, BlenderbotTokenizer,GPT2Tokenizer,BertTokenizer
from transformers import BlenderbotForConditionalGeneration,BlenderbotConfig

import sys
#sys.path.append('../Ubuntu/')
from retrival_model.ret_model import LitESIM, LitBERT, LitDMI
from model import LitS2S
# from Seq2Seq_model import S2S_model
from eval_deb import DEB
# ================================================


# Seed RNGs
import random
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def cmdline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-eckpt", "--esim_checkpoint_path", type=str, required=True, help="Path to ESIM-R3 checkpoint.")
    parser.add_argument("-bckpt", "--bert_checkpoint_path", type=str, required=True, help="Path to BERT-R3 checkpoint.")
    parser.add_argument("-dckpt", "--dmi_checkpoint_path", type=str, required=False, help="Path to DMI checkpoint.")
    parser.add_argument("-dpr", "--data_path_root", type=str, default="./data")
    parser.add_argument("-ds", "--dataset", default="dd", choices=["dstc7", "dd", "dd_cc", "r1M_cc_dd"], help="Choose a dataset.")
    parser.add_argument("-rjp", "--reference_json_path", type=str, required=True, help="Path to reference json file containing ground truth.")
    parser.add_argument("-pjp", "--prediction_json_path", type=str, required=True, help="Directory/Path to input json file(s) containing responses to be graded.")
    parser.add_argument("-gp", "--glove_path", type=str, default="data/glove.txt", help="Path to GloVe embeddings.")
    parser.add_argument("-csv", "--csv_tag", type=str, required=True, help="Path to save the output result(.csv) = `runs/results/{dataset}-results-{tag}.csv")
    return parser.parse_args()

args = cmdline_args()

# 
# ### CUDA and MAX_LEN

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"# Using device: {device}")

# %env CUDA_VISIBLE_DEVICES='0'

C_MAX_LEN = 300
R_MAX_LEN = 30

# COMPARE FORMATTING IN ALL FILES!
def merge_jsons(jsons_path):
    
    pfs = glob.glob(f"{jsons_path}/*.json")
    # pfs = [
    #     "Results/sampled/sampled_model_responses_dd.json",
    #     'Results/sampled/dd_exp1_test.pred.txt.json',
    #     'Results/sampled/outputs/DialoGPT-medium_sample_predictions.json',
    #     'Results/sampled/outputs/DialoGPT-small_sample_predictions.json',
    #     'Results/sampled/outputs/blenderbot-3B_sample_predictions.json',
    #     'Results/sampled/outputs/blenderbot_small-90M_sample_predictions.json',
    #     'Results/sampled/outputs/blenderbot-400M-distill_sample_predictions.json',
    #     "Results/sampled/sampled_model_responses_dd_CE.json",
    #     "Results/sampled/model_responses_dgpt_new.json",
    #     "Results/sampled/model_responses_drpt_new.json",
    # ]
    loaded_pfs = []
    
    x = random.randint(0, 300)
    print(x)

    for pf in pfs:
        if "merged_responses.json" in pf:
            print("Ignoring merged_responses.json")
            continue
        
        with open(pf) as f:
            preds = json.load(f)
            loaded_pfs.append(preds)        
            
            print(f"===========================\nFILE: {pf}")
            print(f"Number of entries found: {len(preds)}\n-----------------------")
            pprint(preds[x])

    # Assertion: all files must have same number of test samples
    assert len(set([len(x) for x in loaded_pfs])) == 1, f"Not of same length {[len(x) for x in loaded_pfs]}"

    # First json is from ESIM/R3 code.
    merged_preds = []
    for entries in zip(*loaded_pfs):
        # pprint(entries)
        root = entries[0]
        others = entries[1:]
        for another in others:
            if 'filename' in another:
                del another['filename']
            if 'con' in another:
                del another ['con']
            if 'gt' in another:
                del another ['gt']
            assert len(another) == 1
            root.update(another)
            # pprint(root)
        # break
        merged_preds.append(root)

    print(f"Source Keys: {list(root.keys())}")
    print(f"Merged samples: {len(merged_preds)}")
    with open(f"{jsons_path}/merged_responses.json", "w") as f:
        json.dump(merged_preds, f, indent=2)
        
pjp = args.prediction_json_path

# assert os.path.exists(pjp), "Predictions path not found"

# if os.path.isdir(pjp):
#     print("Recieved directory path. Merging json files.", pjp)
#     merge_jsons(pjp)
#     args.prediction_json_path = os.path.join(pjp, "merged_responses.json")

print("Final json path:", args.prediction_json_path)

# ## Create Results Table

# 'esim': {}
global_results = defaultdict(dict)


def get_results_df():
    df = pd.DataFrame.from_dict(global_results, orient='index')
    df = df.reset_index()
    df = df.rename({'index': 'Model'}, axis=1)
    return df


#========== INPUT DATA FORMAT ==========
# {'context': '', 'gt': '', 'ESIM-gen': '', 'BERT-gen': ''}
# context: A single string. [SEP] between utterances.
# gt: A single string. No special tokens
# [any model]: A single string. No special tokens

# We preprocess all string inputs to add the BOS/CLS token.
#=======================================

class JsonData(BaseDataClass):
    # Class for reading ground truth from the generated json files
    def __init__(self, json_path, tokenizer):
        self.json_path = json_path

        self.prep_tokenizer_info(tokenizer, C_MAX_LEN, R_MAX_LEN)

        self.data = []
        
        with open(self.json_path, 'r') as f:
            raw_data = json.load(f)

        model_key = "gt"
        for d in raw_data:
            # Proper BOS/CLS is added later.
            ctx = d['con'].replace("[CLS] ", "").strip()
            resp = d[model_key]
            self.data.append((ctx, resp))

    def __getitem__(self, idx):
        ctx_s, resp_s = self.data[idx]
        ctx, resp = self._preprocess(ctx_s, resp_s)
        return {
            "premise": ctx,
            "hypothesis": resp,
            "premise_length": len(ctx),
            "hypothesis_length": len(resp),
            "label": -1,
            "index": -1,
            "context_str": ctx_s,
            "resp_str": resp_s
        }

# Load Ground Truth Data
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
test_dataset = JsonData(args.reference_json_path, bert_tokenizer)
len(test_dataset)
test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False, collate_fn=test_dataset.collate_fn)

# ### The (global) resps array

# Get the list of models to be evaluated
resps = json.load(open(args.prediction_json_path,'r'))
resps[0].pop('con')
if 'gt' in resps[0]: 
   resps[0].pop('gt')
model_list = list(resps[0].keys())
if "filename" in model_list:
    model_list.remove("filename")
print("Model List:", model_list)

# Load and clean/remove unnecessary info from prediction files
resps = json.load(open(args.prediction_json_path,'r'))
assert len(resps) == len(test_dataloader), "Number of samples in prediction file does not match that of the reference file."

for i in range(len(resps)):
    # Use the Ground Truth from the reference file! So overwrite.
    resps[i]['gt'] = test_dataset.data[i][1]

    # Use context from reference, so delete from elsewhere.
    del resps[i]['con']

# ## Average Length

for model in ['gt'] + model_list:
    arr = []
    for b in tqdm(resps, desc=model):
        arr.append(len(re.split(r"[ ]+", b[model])))
    global_results[model]['Avg. Len'] = sum(arr)/len(arr)


print(get_results_df())

# ## BLEU Score
# TODO: Use gt from the reference json file
method = 'BLEU'
for model in ['gt'] + model_list:
    arr = []
    for b in tqdm(resps, desc=f"{model}-{method}"):
        arr.append(sentence_bleu([b['gt']],b[model]))
    global_results[model][method] = sum(arr)/len(arr)

# ## Meteor Score
# TODO: Use gt from the reference json file
method = 'METEOR'
    
for model in ['gt'] + model_list:
    arr = []
    for b in tqdm(resps, desc=f"{model}-{method}"):
        # arr.append(meteor_score([b['gt']],b[model]))
        # need to word tokenize
        arr.append(meteor_score([word_tokenize(b['gt'])], word_tokenize(b[model])))
    global_results[model][method] = sum(arr)/len(arr)

print(get_results_df())

# ## Diversity Score
# TODO: Use gt from the reference json file
method = 'Dist'

for model in ['gt']+model_list:
    unigrams = []
    bigrams = []
    for b in tqdm(resps, desc=f"{model}-{method}"):
        unigrams.extend(word_tokenize(b[model]))
        bigrams.extend(ngrams(word_tokenize(b[model]),2))
    try:
        dist_1 = len(set(unigrams))/len(unigrams)
    except ZeroDivisionError as e:
        print(f"ERROR: {e}")
        dist_1 = 0.

    try:
        dist_2 = len(set(bigrams))/len(bigrams)
    except ZeroDivisionError as e:
        print(f"ERROR: {e}")
        dist_2 = 0.
    
    global_results[model][f"{method}-1"] = dist_1
    global_results[model][f"{method}-2"] = dist_2

print(get_results_df())

## BERT Score

from bert_score import score
import json

with open("/home/bishals/BTP_ULL/results/dd_cc_mle_20.json", "r") as json_file:
    dataset = json.load(json_file)
    
references = []
hypotheses = []

    
for item in dataset:
    references.append(item["gt"])
    hypotheses.append(item["dd_cc_mle_20"])

# Calculate BERTScore
P, R, F1 = score(hypotheses, references, lang="en", verbose=True)

# Print the BERTScore results
print("Precision:", P.mean())
print("Recall:", R.mean())
print("F1 Score:", F1.mean())




# ## MAUDE (ESIM-R3)

# def generate_reward(r3, premise, premise_length, resp, resp_length):
#     # Make sure that CLS/BOS is added to premise and resp -- ESIM expects that 
#     # New dataset class always adds BOS/CLS to the begining of the context/resp
#     # print(premises.shape,hypothesis.shape,premises_lengths.shape,hypothesis_lengths.shape)
#     with torch.no_grad():
#         op = r3(premise, resp)
#         print(op)
#         reward = probs[:, 1]
#         neg_reward = -1 * reward

#         return neg_reward.detach()

# def calc_maude(r3_model, resps, model_list, global_results, metric_name):
#     for model in ['gt']+model_list:
#         gen_resp_scores = []
#         empty_resp = 0
#         for r,b in tqdm(zip(resps,test_dataloader), total=len(resps), desc=f"Analysing {model}"):
#             try:
#                 if r[model] == "":
#                     gs = 0
#                     empty_resp += 1
#                 else:
#                 # gen_resp = bert_tokenizer.tokenize(r[model])
#                 # gen_resp_length = torch.tensor([len(gen_resp)]).to(device)
#                     gen_resp = bert_tokenizer.encode(r[model], return_tensors='pt')
#                     gen_resp_length = torch.tensor([gen_resp.shape[-1]])
#                 # gen_resp_padded = nn.functional.pad(gen_resp, pad=(0, R_MAX_LEN-gen_resp_length), value=bert_tokenizer.pad_token_id)
#                     neg_reward = generate_reward(
#                     r3_model, 
#                     b['premise'].to(device), 
#                     b['premise_length'].to(device),
#                     gen_resp.to(device), 
#                     gen_resp_length.to(device)
#                 )
#                 gen_resp_scores.append(-neg_reward.item())
#             except NotImplementedError as e:
#                 print(e)

#         print(f'MAUDE ({metric_name}) for {model}: {sum(gen_resp_scores)/len(gen_resp_scores)}')
#         global_results[model][f'{metric_name}'] = sum(gen_resp_scores)/len(gen_resp_scores)
    
#     return global_results

# # MAUDE - ESIM R3
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# esim_r3 = LitS2S(vocab_size=len(tokenizer), 
#         pad_idx=tokenizer.pad_token_id,
#         bos_idx=tokenizer.cls_token_id,
#         eos_idx=tokenizer.sep_token_id,
#         num_training_steps=10,
#         run_path_root="runs/",
#         d_model=256, 
#         nhead=8, 
#         num_encoder_layers=6, num_decoder_layers=6, 
#         dim_feedforward=1024, 
#         dropout=0.1,
#         max_pos_idx=1024,
#         learning_rate=0.0001,
#         valuefn_learning_rate=0.001,
#         use_lr_scheduler=False,
#         sampled_reward_mode=False,
#         margin=0.1,
#         use_baseline=False,
#         vanilla_ce=False,
#         equal_reward_alloc=False,
#         enc_weight_init=None,
#         dec_weight_init=None,
#         dataset="dd",
#         data_path_root="data/",
#         retrieval_model_type="ESIM",
#         batch_size=2,
#         epochs=10,
#         early_stopping_patience=5,
#         reddit_steps_per_epoch=500000,
#         device_count=1,
#         no_early_stopping=False,
#         prob_positive=0.9,
#         response_sampler=None,
#         vocab="bert",
#         max_ctx_len=100,
#         max_resp_len=50)
# esim_r3.load_state_dict(torch.load(args.esim_checkpoint_path))
# esim_r3.freeze()
# esim_r3.to(device)

# calc_maude(esim_r3, resps, model_list, global_results, "ESIM-R3")
# print(get_results_df())

# MAUDE - BERT R3
# bert_r3 = LitBERT.load_from_checkpoint(args.bert_checkpoint_path)
# bert_r3.freeze()
# bert_r3.to(device)

# calc_maude(bert_r3, resps, model_list, global_results, "BERT-R3")
# df = get_results_df()

# MAUDE - DMI
# dmi_r3 = LitDMI.load_from_checkpoint(args.dmi_checkpoint_path)
# dmi_r3.freeze()
# dmi_r3.to(device)

# calc_maude(dmi_r3, resps, model_list, global_results, "DMI-R3")
# df = get_results_df()

# df.to_csv("results.csv", index=False)

# print(df)


# ## DEB score

class SimpleCRPairDataset(Dataset):
    def __init__(self, resps, test_dataset, model):
        self.resps = [r[model] for r in resps]
        self.contexts = [b['context_str'] for b in test_dataset]

    def __len__(self):
        return len(self.resps)

    def __getitem__(self, idx):
        return self.contexts[idx], self.resps[idx]

# ### DEB-random-only
deb = DEB(device, "./data/deb_model", is_deb_adversarial=False)
method = 'DEB(r)'

# Sending batch of contexts and responses to deb.evaluate
# Return predicted_labels and valid_resp_prob. Retain mean of both in results

for model in ['gt']+model_list:
    batch_size = 16
    valid_resp_prob = []
    predicted_labels = []
    # for i in tqdm(range(0, len(resps), batch_size), desc=f"{model}-{method}"):
    cr_dataset = SimpleCRPairDataset(resps, test_dataset, model)
    cr_dataloader = DataLoader(cr_dataset, batch_size=batch_size, shuffle=False)

    for contexts, responses in tqdm(cr_dataloader, desc=f"Analyzing {model}-{method}"):
        try:
            p_labels, v_resp_prob = deb.evaluate(contexts, responses)
            valid_resp_prob.extend(v_resp_prob)
            predicted_labels.extend(p_labels)
        except Exception as e:
            print(e)
            continue

    global_results[model][f"{method}-prob"] = sum(valid_resp_prob)/len(valid_resp_prob)
    # global_results[model][f"{method}-percent"] = sum(predicted_labels)/len(predicted_labels)

# ## DEB-random-and-adversarial
deb = DEB(device, "./data/deb_model", is_deb_adversarial=True)
method = 'DEB(a)'

for model in ['gt']+model_list:
    batch_size = 16
    valid_resp_prob = []
    predicted_labels = []
    # for i in tqdm(range(0, len(resps), batch_size), desc=f"{model}-{method}"):
    cr_dataset = SimpleCRPairDataset(resps, test_dataset, model)
    cr_dataloader = DataLoader(cr_dataset, batch_size=batch_size, shuffle=False)

    for contexts, responses in tqdm(cr_dataloader, desc=f"Analyzing {model}-{method}"):
        try:
            p_labels, v_resp_prob = deb.evaluate(contexts, responses)
            valid_resp_prob.extend(v_resp_prob)
            predicted_labels.extend(p_labels)
        except Exception as e:
            print(e)
            continue

    global_results[model][f"{method}-prob"] = sum(valid_resp_prob)/len(valid_resp_prob)
    # global_results[model][f"{method}-percent"] = sum(predicted_labels)/len(predicted_labels)


# # Embedding Similarity

# embeddings_dict = {}
# with open(args.glove_path, 'r') as f:
#     for line in tqdm(f):
#         # try:
#         values = line.split()
#         word = values[0]
#         vector = np.asarray(values[1:], "float32")
#         embeddings_dict[word] = vector
#         # except ValueError:
#         #     print(values)

# embeddings_dict['<UNK>'] = np.asarray([0]*300,"float32")
            

# # ## Average

# method = 'ES(avg)'

# for model in model_list:
#     arr = []
#     for b in tqdm(resps, desc=f"{model}-{method}"):

#         dy0 = b['gt']
#         dy1 = b[model]

#         gt_embs = []
#         for t in word_tokenize(dy0):
#             try:
#                 gt_embs.append(embeddings_dict[t])
#             except KeyError:
#                 continue
#                 #gt_embs.append(embeddings_dict['<UNK>'])
#         gt_embs = np.array(gt_embs)    
#         gt_avg_emb = gt_embs.mean(axis=0)

#         if( len(word_tokenize(dy1))==0):
#               continue
#         r_embs = []
#         for t in word_tokenize(dy1):
#             try:
#                 r_embs.append(embeddings_dict[t])
#             except KeyError:
#                 continue
#                 #r_embs.append(embeddings_dict['<UNK>'])
        
#         r_embs = np.array(r_embs)
#         r_avg_emb = r_embs.mean(axis=0)
    
#         try:
#             arr.append(np.dot(gt_avg_emb,r_avg_emb)/(np.linalg.norm(gt_avg_emb,ord=2)*np.linalg.norm(r_avg_emb,ord=2)))
#         except:
#             continue
#     try:
#         global_results[model][f"{method}"] = sum(arr)/len(arr)
#     except ZeroDivisionError as e:
#         global_results[model][f"{method}"] = 0

# # ## Extrema

# method = 'ES(extrema)'

# for model in model_list:
#     arr = []
#     for b in tqdm(resps, desc=f"{model}-{method}"):

#         dy = [ b['gt'],b[model]]

#         gt_embs = []
#         for t in word_tokenize(dy[0]):
#             try:
#                 gt_embs.append(embeddings_dict[t])
#             except KeyError:
#                 continue
#                 #gt_embs.append(embeddings_dict['<UNK>'])
#         gt_embs = np.array(gt_embs)
#         if (gt_embs.shape[0]==0): 
#             continue
#         gt_ext_idx = np.argmax(abs(gt_embs),axis=0)
#         gt_ext_emb = np.zeros(300)
#         for j,i in enumerate(gt_ext_idx):
#             gt_ext_emb[j] = gt_embs[i,j] 


#         r_embs = []
#         for t in word_tokenize(dy[1]):
#             try:
#                 r_embs.append(embeddings_dict[t])
#             except KeyError:
#                 continue
#                 #r_embs.append(embeddings_dict['<UNK>'])

#         r_embs = np.array(r_embs)
#         if (r_embs.shape[0]==0): 
#             continue
#         r_ext_idx = np.argmax(abs(r_embs),axis=0)
#         r_ext_emb = np.zeros(300)
#         for j,i in enumerate(r_ext_idx):
#             r_ext_emb[j] = r_embs[i,j] 

#         try:
#             arr.append(np.dot(gt_ext_emb,r_ext_emb)/(np.linalg.norm(gt_ext_emb,ord=2)*np.linalg.norm(r_ext_emb,ord=2)))
#         except:
#             print("error")
#             continue
#     try:
#         global_results[model][f"{method}"] = sum(arr)/len(arr)
#     except ZeroDivisionError:
#         global_results[model][f"{method}"] = 0

# # ## Greedy

# method = 'ES(greedy)'

# for model in model_list:
    
#     arr = []
#     for b in tqdm(resps, desc=f"{model}-{method}"):

#         dy0 = b['gt']
#         dy1 = b[model]

#         gt_embs = []
#         for t in word_tokenize(dy0):
#             try:
#                 gt_embs.append(embeddings_dict[t])
#             except KeyError:
#                 continue
#                 #gt_embs.append(embeddings_dict['<UNK>'])
#         gt_embs = np.array(gt_embs)
#         if (gt_embs.shape[0]==0): continue

#         r_embs = []
#         for t in word_tokenize(dy1):
#             try:
#                 r_embs.append(embeddings_dict[t])
#             except KeyError:
#                 continue
#                 #r_embs.append(embeddings_dict['<UNK>'])

#         r_embs = np.array(r_embs)
#         if (r_embs.shape[0]==0): continue

#         mat = np.zeros([r_embs.shape[0],gt_embs.shape[0]])#np.matmul(r_embs,gt_embs.transpose([1,0]))
#         for j in range(r_embs.shape[0]):
#             for k in range(gt_embs.shape[0]):
#                 mat[j,k] = r_embs[j].dot(gt_embs[k])/(np.linalg.norm(r_embs[j])*np.linalg.norm(gt_embs[k]))
#         r_gt = np.sum(mat.max(axis=1))/(r_embs.shape[0])
#         gt_r = np.sum(mat.max(axis=0))/(gt_embs.shape[0])

#         try:
#             arr.append((r_gt + gt_r)/2)
#         except:
#             continue

#     try:
#         global_results[model][f"{method}"] = sum(arr)/len(arr)
#     except ZeroDivisionError:
#         global_results[model][f"{method}"] = 0

# # Save results Table

df = get_results_df()
print(df.head())
os.makedirs("runs/results", exist_ok=True)
r_csv_path = f'./runs/results/{args.dataset}-results-{args.csv_tag}.csv'
df.to_csv(r_csv_path, index=False)