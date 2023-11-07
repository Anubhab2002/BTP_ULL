import torch
from model import LitS2S
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


# Load your pre-trained model checkpoint and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model_checkpoint = "/home/bishals/BTP_ULL/models/mle_dialogue_model_20.pth" 
model = LitS2S(vocab_size=len(tokenizer), 
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
model.load_state_dict(torch.load(model_checkpoint))
# model.to(device)

with open("/home/bishals/BTP_ULL/ablation_set1/dd_cc_BERT_D1295001.json", "r") as json_file:  
    dataset = json.load(json_file)
    
results = []
x = 0
# Generate responses for the dataset
for item in dataset:
    context = item["con"]
    ground_truth = item["gt"]

    # Tokenize the context input
    input_ids = tokenizer(context, return_tensors="pt", padding=True, truncation=True)
    
    input_ids = input_ids['input_ids']

    # Generate a response using the model
    with torch.no_grad():
        output = model(input_ids, max_decode_len=64)
        
    # print(output)

    # Decode the generated response
    generated_response = tokenizer.decode(output["sequence"][0], skip_special_tokens=True)
    
    results.append(
        {
            "con": context,
            "gt": ground_truth,
            "dd_cc_mle_20": generated_response,
        }
    )
    
    if x%100==0: 
        print(x)
        
    if x == 300:
        break
    
    x = x+1

# Save the results to a JSON file
with open("/home/bishals/BTP_ULL/results/dd_cc_mle_20.json", "w") as json_file:
    json.dump(results, json_file, indent=4)
    
    
# from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
# import json

# # Load the pre-trained T5 model and tokenizer
# model_name = "t5-small"
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Create a pipeline for text-to-text generation
# seq2seq_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# # Load the dataset from a JSON file
# with open("/home/bishals/BTP_ULL/ablation_set1/dd_cc_300.json", "r") as json_file:  # Replace with your dataset file path
#     dataset = json.load(json_file)

# # Create an empty list to store the results
# results = []

# # Generate responses for the dataset
# for item in dataset:
#     context = item["con"]
#     ground_truth = item["gt"]
#     # Generate a response using the pipeline
#     generated_response = seq2seq_pipeline(context, max_length=100, do_sample=True)

#     # Append the results to the list
#     results.append({
#         "con": context,
#         "gt": ground_truth,
#         "generated_response": generated_response[0]["generated_text"],
#     })

# # Save the results to a JSON file
# with open("/home/bishals/BTP_ULL/results/dd_cc_mle_20.json", "w") as json_file:
#     json.dump(results, json_file, indent=4)
