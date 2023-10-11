import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import os
import random
import math
import time
import glob

from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BlenderbotForConditionalGeneration
from transformers.models.blenderbot.modeling_blenderbot import BlenderbotLearnedPositionalEmbedding
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# from utils import top_k_top_p_filtering
# from retrieval_model.model import LitESIM, LitBERT, LitDMI, LitDMIZero

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=100):
        super().__init__()

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = np.sqrt(hid_dim)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).expand(batch_size, -1).to(src.device)
        # print("pos: ", pos.shape)
        # pos = [batch size, src len]

        token_ = self.tok_embedding(src)
        pos_ = self.pos_embedding(pos)

        src = self.dropout((token_ * self.scale)+pos_)

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]

        return src
    
    def init_roberta(self, roberta_name: str):
        print("\x1b[31;1minside init_roberta\x1b[0m")
        if "base" in roberta_name:
            dropout, n_heads, pf_dim = 0.1, 12, 3072
            max_length, input_dim = 512+2, 50265
            n_layers, hid_dim = 12, 768
        if roberta_name.lower().startswith("dmi"):
            from retrieval_model.dmi import SMI
            rob = SMI.from_pretrained(
                roberta_name, roberta_init=True, 
                roberta_name="roberta-base",
            ).encoder
        elif roberta_name.startswith("roberta"):
            from transformers import RobertaModel
            rob = RobertaModel.from_pretrained(roberta_name)
        else: 
            err_msg = f"No initialization strategy for {roberta_name}. Currently supported: roberta-base (roberta base init), dmi* (DMI base init)"
            raise Exception(err_msg)
        # resize token embedding layer:
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        state_dict = rob.embeddings.word_embeddings.state_dict()
        print("self.tok_embedding init:", self.tok_embedding.load_state_dict(state_dict))
        # resize position embedding layer:
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        state_dict = rob.embeddings.position_embeddings.state_dict()
        print("self.pos_embedding init:", self.pos_embedding.load_state_dict(state_dict))
        # resize transformer layers.
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        for i, layer in enumerate(self.layers):
            rob_layer = rob.encoder.layer[i]
            # attention.
            rob_attn = rob_layer.attention
            rob_self = rob_attn.self
            rob_out = rob_attn.output
            # intermediate.
            rob_inter = rob_layer.intermediate
            # output.
            rob_out_layer = rob_layer.output
            # match key-query-value layers.
            state_dict = rob_self.key.state_dict()
            print(f"self.layer[{i}].self_attention.fc_k init", 
                  layer.self_attention.fc_k.load_state_dict(state_dict))
            state_dict = rob_self.query.state_dict()
            print(f"self.layer[{i}].self_attention.fc_q init", 
                  layer.self_attention.fc_q.load_state_dict(state_dict))
            state_dict = rob_self.value.state_dict()
            print(f"self.layer[{i}].self_attention.fc_v init", 
                  layer.self_attention.fc_v.load_state_dict(state_dict))
            # match self attention output layer weights.
            state_dict = rob_out.dense.state_dict()
            print(f"self.layer[{i}].self_attention.fc_o init",
                  layer.self_attention.fc_o.load_state_dict(state_dict))
            state_dict = rob_out.LayerNorm.state_dict()
            print(f"self.layer[{i}].self_attn_layer_norm init",
                  layer.self_attn_layer_norm.load_state_dict(state_dict))
            # match feedforward layers.
            state_dict = rob_inter.dense.state_dict()
            print(f"self.layer[{i}].positionwise_feedforward.fc_1 init", 
                  layer.positionwise_feedforward.fc_1.load_state_dict(state_dict))
            state_dict = rob_out_layer.dense.state_dict()
            print(f"self.layer[{i}].positionwise_feedforward.fc_2 init", 
                  layer.positionwise_feedforward.fc_2.load_state_dict(state_dict))
            state_dict = rob_out_layer.LayerNorm.state_dict()
            print(f"self.layer[{i}].ff_layer_norm init", 
                  layer.ff_layer_norm.load_state_dict(state_dict))
        print("\x1b[32;1minit_roberta successful\x1b[0m")

class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = np.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=100):
        super().__init__()

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = np.sqrt(hid_dim)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).expand(batch_size, -1).to(trg.device)

        # pos = [batch size, trg len]

        pos_ = self.pos_embedding(pos)
        trg = self.dropout((self.tok_embedding(trg) * self.scale)+pos_)

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        # output = [batch size, trg len, output dim]

        return output, attention


    def init_roberta(self, roberta_name: str):
        print("\x1b[31;1minside init_roberta\x1b[0m")
        if "base" in roberta_name:
            dropout, n_heads, pf_dim = 0.1, 12, 3072
            max_length, output_dim = 512+2, 50265
            n_layers, hid_dim = 12, 768
        # resize fc_out (vocabulary output) layer.
        self.fc_out = nn.Linear(hid_dim, output_dim)
        if roberta_name.lower().startswith("dmi"):
            from retrieval_model.dmi import SMI
            rob = SMI.from_pretrained(
                roberta_name, roberta_init=True, 
                roberta_name="roberta-base",
            ).encoder
        elif roberta_name.startswith("roberta"):
            from transformers import RobertaModel
            rob = RobertaModel.from_pretrained(roberta_name)
        else: 
            err_msg = f"No init strat for {roberta_name}. Supported strats: [roberta*base*, dmi*base*]"
            raise Exception(err_msg)
        # resize token embedding layer:
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        state_dict = rob.embeddings.word_embeddings.state_dict()
        print("self.tok_embedding init:", self.tok_embedding.load_state_dict(state_dict))
        # resize position embedding layer:
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        state_dict = rob.embeddings.position_embeddings.state_dict()
        print("self.pos_embedding init:", self.pos_embedding.load_state_dict(state_dict))
        # resize transformer layers.
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        for i, layer in enumerate(self.layers):
            rob_layer = rob.encoder.layer[i]
            # attention.
            rob_attn = rob_layer.attention
            rob_self = rob_attn.self
            rob_out = rob_attn.output
            # intermediate.
            rob_inter = rob_layer.intermediate
            # output.
            rob_out_layer = rob_layer.output
            # match key-query-value layers.
            state_dict = rob_self.key.state_dict()
            print(f"self.layer[{i}].self_attention.fc_k init", 
                  layer.self_attention.fc_k.load_state_dict(state_dict))
            state_dict = rob_self.query.state_dict()
            layer.self_attention.fc_q.load_state_dict(state_dict)
            print(f"self.layer[{i}].self_attention.fc_q init", 
                  layer.self_attention.fc_q.load_state_dict(state_dict))
            state_dict = rob_self.value.state_dict()
            layer.self_attention.fc_v.load_state_dict(state_dict)
            print(f"self.layer[{i}].self_attention.fc_v init", 
                  layer.self_attention.fc_v.load_state_dict(state_dict))
            # match self attention output layer weights.
            state_dict = rob_out.dense.state_dict()
            print(f"self.layer[{i}].self_attention.fc_o init",
                  layer.self_attention.fc_o.load_state_dict(state_dict))
            state_dict = rob_out.LayerNorm.state_dict()
            print(f"self.layer[{i}].self_attn_layer_norm init",
                  layer.self_attn_layer_norm.load_state_dict(state_dict))
            # match feedforward layers.
            state_dict = rob_inter.dense.state_dict()
            print(f"self.layer[{i}].positionwise_feedforward.fc_1 init", 
                  layer.positionwise_feedforward.fc_1.load_state_dict(state_dict))
            state_dict = rob_out_layer.dense.state_dict()
            print(f"self.layer[{i}].positionwise_feedforward.fc_2 init", 
                  layer.positionwise_feedforward.fc_2.load_state_dict(state_dict))
            state_dict = rob_out_layer.LayerNorm.state_dict()
            print(f"self.layer[{i}].ff_layer_norm init", 
                  layer.ff_layer_norm.load_state_dict(state_dict))
            
            # TODO: Components not initialized yet.
            # enc_attn_layer_norm
            # encoder_attention
            # Initialization for lm_head for roberta??
        print("\x1b[32;1minit_roberta successful\x1b[0m")

class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention

class BlenderbotEncoder(nn.Module):
    def __init__(self, hf_blenderbot_instance):
        super().__init__()
        
        self.encoder = hf_blenderbot_instance.model.encoder        
        
    def forward(self, src, src_mask):
        # TODO: We need to check if this mask format is suitable for blenderbot
        # src_mask = (src_mask.squeeze(1).squeeze(1))*1
        src_mask = src_mask.squeeze(1).squeeze(1).to(torch.int)
        # print(src_mask)
        enc_src = self.encoder(input_ids = src, attention_mask = src_mask, return_dict=True)
        return enc_src.last_hidden_state
    
class BlenderbotDecoder(nn.Module):
    def __init__(self, hf_blenderbot_instance):
        super().__init__()
        
        self.decoder = hf_blenderbot_instance.model.decoder
        self.fc_out = hf_blenderbot_instance.lm_head
        # self.final_logits_bias = hf_blenderbot_instance.final_logits_bias
        self.register_buffer("final_logits_bias", hf_blenderbot_instance.final_logits_bias)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        src_mask = src_mask.squeeze(1).squeeze(1).to(torch.int)
        trg_mask = trg_mask.squeeze(1)[:,-1,:].to(torch.int)
        # print(trg_mask)
        # print(trg)
        decoder_out = self.decoder(input_ids=trg, attention_mask=trg_mask,\
                                   encoder_hidden_states=enc_src, encoder_attention_mask=src_mask,\
                                   output_attentions=True, return_dict=True)
        # attention = [batch size, n heads, trg len, src len]
        # print(decoder_out.keys())
        enc_trg = decoder_out.last_hidden_state
        
        ## TODO: Verify whether this is the cross-attn from the final layer.
        attention = decoder_out.cross_attentions[0]
        
        # output = [batch size, trg len, V]
        output = self.fc_out(enc_trg) + self.final_logits_bias

        return output, attention
    

def resize_blenderbot_positional_embeddings(max_position, hf_blenderbot_instance):
    assert hf_blenderbot_instance.model.encoder.config.max_position_embeddings < max_position, "You can only increase the size of enc positional embedding matrix."
    assert hf_blenderbot_instance.model.decoder.config.max_position_embeddings < max_position, "You can only increase the size of dec positional embedding matrix."
    
    # update config
    hf_blenderbot_instance.model.encoder.config.max_position_embeddings = max_position
    hf_blenderbot_instance.model.decoder.config.max_position_embeddings = max_position
    
    # Encoder: create the new weight matrix
    og_weights = hf_blenderbot_instance.model.encoder.embed_positions.weight.detach()
    n,d = og_weights.shape
    extension = torch.rand((max_position - n,d)).to(og_weights)
    new_weights = torch.cat([og_weights, extension], 0)

    pe_enc = BlenderbotLearnedPositionalEmbedding(max_position, d)
    pe_enc.load_state_dict({
        "weight": new_weights
    })

    # Decoder: create the new weight matrix
    og_weights = hf_blenderbot_instance.model.decoder.embed_positions.weight.detach()
    n,d = og_weights.shape
    extension = torch.rand((max_position - n,d)).to(og_weights)
    new_weights = torch.cat([og_weights, extension], 0)

    pe_dec = BlenderbotLearnedPositionalEmbedding(max_position, d)
    pe_dec.load_state_dict({
        "weight": new_weights
    })
    
    #     hf_blenderbot_instance.model.encoder.embed_positions = BlenderbotLearnedPositionalEmbedding.from_pretrained(new_weights)
    return pe_enc, pe_dec

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        
        Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
    
    pred_token = torch.multinomial(F.softmax(logits, -1), 1) # [BATCH_SIZE, 1]
    return pred_token


class LitS2S(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitS2S")
        parser.add_argument("-d",  "--d_model", type=int, default=512, help="Size of representations in transformer")
        parser.add_argument("-nh", "--nhead", type=int, default=8, help="Number of self-attn heads in the transformer")
        parser.add_argument("-el", "--num_encoder_layers", type=int, default=6, help="Number of encoder layers in the transformer")
        parser.add_argument("-dl", "--num_decoder_layers", type=int, default=6, help="Number of decoder layers in the transformer")
        parser.add_argument("-ff", "--dim_feedforward", type=int, default=2048, help="Hidden layer size of feedforward layer in the transformer")
        parser.add_argument("-dr", "--dropout", type=float, default=0.1, help="Dropout rate in the transformer")

        parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate for seq2seq.")
        parser.add_argument("-vflr", "--valuefn_learning_rate", type=float, default=0.001, help="Learning rate for state-value function approximation.")
        # parser.add_argument("-scdl", "--use_lr_scheduler", action="store_true", help="Use learning rate scheduler.")

        parser.add_argument("-ce", "--vanilla_ce", action="store_true", help="Use vanilla cross entropy loss for training.")
        parser.add_argument("-srm", "--sampled_reward_mode", action="store_true", help="Use sampled output for reward calculation.")
        parser.add_argument("-m", "--margin", default=0.1, type=float, help="Margin for reward")

        # reward baseline
        parser.add_argument("-ub", "--use_baseline", action="store_true", help="Use reward baseline.")

        parser.add_argument("-eq", "--equal_reward_alloc", action="store_true", help="Allocate equal reward for all tokens.")
        parser.add_argument("-ewi", "--enc_weight_init", type=str, 
                       help="<roberta model name>/<dmi path>/'blender' to be used.")
        parser.add_argument("-dwi", "--dec_weight_init", type=str, 
                       help="<roberta model name>/<dmi path>/'blender' to be used.")
        return parent_parser


    def __init__(
        self, 
        vocab_size, 
        pad_idx,
        bos_idx,
        eos_idx,
        num_training_steps,
        run_path_root="runs/",
        d_model=256, 
        nhead=8, 
        num_encoder_layers=6, num_decoder_layers=6, 
        dim_feedforward=1024, 
        dropout=0.1,
        max_pos_idx=256,
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
        max_ctx_len=150,
        max_resp_len=60):
        super().__init__()

        # Tokenization Configuration
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        
        assert self.pad_idx is not None, "PAD cannot be None"
        assert self.bos_idx is not None, "BOS cannot be None"
        assert self.eos_idx is not None, "EOS cannot be None"

        # Model Configuration
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_pos_idx = max_pos_idx
        self.retrieval_model_type = retrieval_model_type
        
        # Training Configuration
        self.sampled_reward_mode = sampled_reward_mode
        self.margin = margin
        self.vanilla_ce = vanilla_ce
        self.equal_reward_alloc = equal_reward_alloc
        self.learning_rate = learning_rate
        self.use_baseline = use_baseline
        self.valuefn_learning_rate = valuefn_learning_rate
        self.use_lr_scheduler = use_lr_scheduler
        self.num_training_steps = num_training_steps

        self.enc_weight_init = enc_weight_init
        self.dec_weight_init = dec_weight_init

        # R3
        self.run_path_root = run_path_root

        self.save_hyperparameters()

        if enc_weight_init == "blender" or dec_weight_init == "blender":
            self.hf_blenderbot = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
            self.hf_blenderbot.resize_token_embeddings(self.vocab_size)
            
            self.hf_blenderbot.model.encoder.embed_positions, self.hf_blenderbot.model.decoder.embed_positions = \
                    resize_blenderbot_positional_embeddings(self.max_pos_idx, self.hf_blenderbot)
            
            self.encoder = BlenderbotEncoder(self.hf_blenderbot)
            self.decoder = BlenderbotDecoder(self.hf_blenderbot)
        else:
            self.encoder = Encoder(
                vocab_size, self.d_model, 
                self.num_encoder_layers, self.nhead, self.dim_feedforward,
                self.dropout, max_length=self.max_pos_idx)
            if enc_weight_init is not None:
                self.encoder.init_roberta(enc_weight_init)

            self.decoder = Decoder(
                vocab_size, self.d_model, self.num_decoder_layers, self.nhead, self.dim_feedforward,
                self.dropout, max_length=self.max_pos_idx)
            if dec_weight_init is not None:
                self.decoder.init_roberta(dec_weight_init)

        # Baseline function
        layers = nn.ModuleList()
        vf_ff_dims = [self.d_model, 2048] # hidden layers in valuefn
        
        for i in range(len(vf_ff_dims)):
            layers.append(nn.Dropout(p=self.dropout))
            if i<len(vf_ff_dims)-1:
                layers.append(nn.Linear(vf_ff_dims[i],vf_ff_dims[i+1]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(vf_ff_dims[i],1))
        layers.append(nn.Sigmoid())
        self.value_net = nn.Sequential(*layers)
        self.mse_loss = nn.MSELoss()
        # self.valuefn = d (encoder cls) -> 1
        # Loss equivalent to update eqn in Sutton & Barto (2nd ed. 2020, p330)
        # delta*valuefn(context) or delta^T@delta, which is nothing but the MSE loss
        # delta = G - valuefn(context) = R3(c,r) - valuefn(c)

        # Ready reward model R3
        # self.load_pretrained_r3()    
    '''
    def load_pretrained_r3(self):
        if self.retrieval_model_type == "DMIZ_Medium": 
            self.r3_model = LitDMIZero('data/DMI_Medium/model_current.pth', 1, self.pad_idx,
                retrieval_model_type=self.retrieval_model_type,
                roberta_init=True,
                roberta_name="google/bert_uncased_L-8_H-768_A-12")
        elif self.retrieval_model_type == "DMIZ_BASE":
            self.r3_model = LitDMIZero('data/DMI_Base/model_current.pth', 1, self.pad_idx,
                retrieval_model_type=self.retrieval_model_type,
                roberta_init=True,
                roberta_name="roberta-base")
        else:
            # R3: Load ESIM checkpoint
            # Model Checkpoint
            search_path = os.path.join(self.run_path_root, f"{self.retrieval_model_type}/")
            if self.vocab == "blender":
                search_path = os.path.join(search_path, "blender/")
            ckpt_files = glob.glob(search_path + "*.ckpt")
                
            choice = ckpt_files[0]
            print(f"Loading {self.retrieval_model_type} pretrained model from {choice}")
            # print("Loading ESIM checkpoint from: {}".format(choice))
            
            if self.retrieval_model_type == "ESIM":
                self.r3_model = LitESIM.load_from_checkpoint(choice)
            elif self.retrieval_model_type == "BERT":
                self.r3_model = LitBERT.load_from_checkpoint(choice)
            elif self.retrieval_model_type in ["DMI", "DMI_BASE"]:
                self.r3_model = LitDMI.load_from_checkpoint(choice)
        
        # Freeze it!
        self.r3_model.freeze()
    '''
    def generate_reward(self, premise, premise_length, resp, resp_length):
        # Make sure that CLS/BOS is added to premise and resp -- ESIM expects that 
        # New dataset class always adds BOS/CLS to the begining of the context/resp
        # print(premises.shape,hypothesis.shape,premises_lengths.shape,hypothesis_lengths.shape)
        with torch.no_grad():
            logits, probs, attn_vec = self.r3_model(premise, premise_length, resp, resp_length)
            reward = probs[:, 1]
            neg_reward = -1 * reward

            return neg_reward.detach(), attn_vec.detach()

    def make_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len)).to(trg.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg=None, max_decode_len=None, temperature = 1.0, epsilon=0.0, min_length = 6, top_p = 0.5, top_k = 0):
        # If trg is NOT None -> teacher forcing -> trg is input to the decoder
        # trg is None for inference
        # If trg is None, we do a greedy/topk/top-p decoding

        src_mask = self.make_src_mask(src) # function to create attention mask for source sequence
        enc_src = self.encoder(src, src_mask) # pass the seq and the attention mask to the encoder

        output = {} # autoregressive decoding output (comparable to the contiguous logits part ig...)

        output["encoder_hidden_states"] = enc_src # IDK why

        if trg is not None: # token level loss calculation part
            trg_mask = self.make_trg_mask(trg) 
            out_logits, _ = self.decoder(trg, enc_src, trg_mask, src_mask)
            # print(out_logits.shape)

            # output['sequence'] = torch.LongTensor(trg_indexes[1:]).unsqueeze(0).to(device)
            output['scores'] = F.log_softmax(out_logits, dim=-1)
            output['logits'] = out_logits
        else: # sequence level loss calculation part (We are doing THIS)
            # Greedy/Sampling/Top-k/Top-p decoding
            # Configuration
            # epsilon = 0 is complete sampling
            # epsilon = 1 is complete greedy
            # epsilon = -1 is top_k + top_p sampling
            assert max_decode_len is not None, "max_decode_len must be provided for inference" # completion seq max length must be defined
            
            current_batch_size = src.shape[0] # src is [batch_size, _]

            trg_indexes = [self.bos_idx]*current_batch_size # initialise the target 

            # INIT
            # logits_list = []
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(src.device) # convert them to tensors
            decreasing_series = torch.arange(max_decode_len, 0, -1).unsqueeze(0).to(src.device) # IDK
            # BOS already added, so max_decode_len - 1
            for i in range(max_decode_len - 1):                
                trg_mask = self.make_trg_mask(trg_tensor)

                out_logits, _ = self.decoder(trg_tensor, enc_src, trg_mask, src_mask) # decode the sequence
                out_logits = out_logits[:, -1] / (temperature) # normalise based on temperature
                # print("SHAPE: ", out_logits.shape)
                if i < min_length:
                    out_logits[:, 0] = -float('Inf')
                    out_logits[:, self.eos_idx] = -float('Inf')
                if i >= 0:
                    out_logits[:, self.bos_idx] = -float('Inf')

                # logits_list.append(out_logits.unsqueeze(1))
                
                if (np.random.rand() < epsilon and epsilon > 0):
                    pred_token = out_logits.argmax(-1).unsqueeze(1)
                elif (np.random.rand() >= epsilon and epsilon >= 0):
                    pred_token = torch.multinomial(F.softmax(out_logits, -1), 1)
                else:
                    pred_token = top_k_top_p_filtering(out_logits, top_k=top_k, top_p=top_p)
                    
                trg_tensor = torch.cat([trg_tensor, pred_token], dim=1) # add the predicted token to the generated sequence

            
            # out_logits = torch.cat(logits_list, dim=1)
            # TODO: Should we remove BOS tokens? I think not, ESIM uses them.
            # trg_tensor = trg_tensor[:, 1:] # Remove BOS
            
            trg_tensor[:, -1] = self.eos_idx # Force atleast one EOS # add EOS token to the end of the sequence generated
            # print(trg_tensor.shape)
            trg_lengths_tensor = torch.argmax(decreasing_series*(trg_tensor == self.eos_idx), dim=1)
            trg_lengths = [1+l for l in trg_lengths_tensor.tolist()] # since argmax is 0 indexed
            gen_max_len = max(trg_lengths)
            trg_tensor = trg_tensor[:, :gen_max_len] # get the target tensor with max_length
            
            # PAD the output beyond eos?
            trg_tensor[(max_decode_len - decreasing_series[:, :gen_max_len]) > trg_lengths_tensor.unsqueeze(1)] = self.pad_idx # add padding at the end

            output['sequence'] = trg_tensor # output sequence contains the target_tensor
            # print("output sequence shape", output['sequence'].shape)
            output['sequence_len'] = torch.LongTensor(trg_lengths)
            output['logits'] = out_logits
            # print("OUTPUT LOGITS SHAPE: ", output['logits'].shape)

        return output
    

    def training_step(self, batch, batch_idx):
        b_premise = batch['premise']
        b_premise_length = batch['premise_length']
        if 'negative_sample' in batch:
            b_hypothesis = batch['negative_sample'].to(batch['hypothesis'])
            b_hypothesis_length = batch['negative_sample_length'].to(batch['hypothesis_length'])
        else:
            b_hypothesis = batch['hypothesis']
            b_hypothesis_length = batch['hypothesis_length']
        b_label = batch['label']

        trg_resp = b_hypothesis[:, :-1]
        y = self(b_premise.squeeze(1), trg_resp)
        score_mask = (trg_resp != 0).unsqueeze(-1).expand(-1, -1, self.vocab_size)

        # gen_resp_arr.append(gen_resp)
        '''
        gen_resp_length = torch.tensor([gen_resp.shape[1]]).to(device)
        if gen_resp_length.item()<R_MAX_LEN:
            padding = torch.tensor([[0]*(R_MAX_LEN-gen_resp_length.item())]).to(device)
            gen_resp = torch.cat((gen_resp,padding),dim=1)
        '''
        if not self.sampled_reward_mode:
            neg_reward, attn_vec = self.generate_reward(b_premise.squeeze(1),
                                                b_premise_length, b_hypothesis,
                                                b_hypothesis_length)
        else:
            # Sample a response first
            greedy_y = y['scores'].argmax(dim=2)
            # TODO: sampled_y = torch.multinomial(y['scores'], num_samples=1) if scores is softmax!
            neg_reward, attn_vec = self.generate_reward(b_premise.squeeze(1),
                                                    b_premise_length, greedy_y,
                                                    b_hypothesis_length)

        # Calculate the reward only for the all samples
        # Since we will incorporate both type of samples now, the average reward should increase!
        original_reward = -neg_reward

        # print(f"REWARD: {reward}\n")
        scores = y['scores'] * score_mask
        
        # R3 or Cross-Entropy Loss?
        if not self.vanilla_ce:
            # Baseline 
            if self.use_baseline:
                # (now margin) value function
                # TODO: Confirm that input data has CLS token!
                context_cls_embeddings = y['encoder_hidden_states'][:, 0, :]
                # print(y['encoder_hidden_states'].shape)
                state_value = self.value_net(context_cls_embeddings.detach()).squeeze()
                valuefn_loss = self.mse_loss(original_reward, state_value) 

                # neg_reward is `-delta` in Sutton Barto's book
                neg_reward = neg_reward + state_value #.detach() -- detached later (L+8)
            else:
                neg_reward = neg_reward + self.margin
            # neg_reward = neg_reward.unsqueeze(-1).repeat(1, R_MAX_LEN).detach()
            BATCH_R_MAX_LEN = b_hypothesis.shape[1] - 1 # first is cls token
            neg_reward = neg_reward.unsqueeze(-1).expand(-1, BATCH_R_MAX_LEN).detach()
            if self.equal_reward_alloc:
                # Allocate equal reward to all non-pad tokens
                # neg_reward = neg_reward/b_hypothesis_length.unsqueeze(1)
                pass
            else:
                # Remove cls from attn_vec
                attn_vec = attn_vec[..., 1:]
                # Extend attn vec with zeros on dim 1 -- To be padded with ZERO! not pad_idx
                attn_vec = F.pad(attn_vec, (0, BATCH_R_MAX_LEN - attn_vec.shape[-1]), value=0)
                neg_reward = neg_reward * attn_vec.mean(1)  # [:len(scores)]
                # alloc_rew.append(neg_reward)
                # neg_reward = neg_reward.flip([-1]).cumsum(dim=-1).flip([-1])
                # print(neg_reward.shape)
                # print(torch.gather(scores,-1,b_hypothesis.unsqueeze(-1).to(device)).shape)

            Seq2Seq_loss = (neg_reward * (
                torch.gather(scores, -1, b_hypothesis[:, 1:].unsqueeze(-1))).squeeze(-1))
        else:
            Seq2Seq_loss = -torch.gather(scores, -1, b_hypothesis[:, 1:].unsqueeze(-1)).squeeze(-1)

        # print(Seq2Seq_loss.shape)
        # Normalize Seq2Seq_loss by the length of the hypothesis
        Seq2Seq_loss = Seq2Seq_loss / b_hypothesis_length.unsqueeze(1).float()
        Seq2Seq_loss = Seq2Seq_loss.sum(-1).mean(0)
        # print(Seq2Seq_loss)
        if self.use_baseline:
            Seq2Seq_loss = Seq2Seq_loss + valuefn_loss
            self.log("train/valuefn_loss", valuefn_loss.item(), sync_dist=True)

        self.log("train/loss", Seq2Seq_loss.item(), sync_dist=True)
        self.log("train/r3", original_reward.mean().item(), sync_dist=True)
        
        return Seq2Seq_loss

    def validation_step(self, batch, batch_idx):
        b_hypothesis = batch['hypothesis']
        b_premise = batch['premise']
        b_hypothesis_length = batch['hypothesis_length']
        b_premise_length = batch['premise_length']
        b_label = batch['label']

        current_batch_size = b_hypothesis.shape[0]

        y = self(b_premise.squeeze(1), max_decode_len=30, epsilon=0)

        gen_resp = y['sequence']
        gen_resp_length = y['sequence_len']
        padding = (gen_resp_length[:, None] < torch.arange(0, gen_resp.shape[1], 1)[None, :].to(gen_resp_length.device))
        gen_resp[padding] = self.pad_idx
        
        # TODO: pad the sampled sequence based on the length of the generated sequence
        # The new, optimized Seq2Seq_generate_valid function doesn't check for padding beyond EOS_ID
        neg_reward, attn_vec = self.generate_reward(b_premise,
                                                b_premise_length, gen_resp, gen_resp_length)
        
        self.log("valid/r3", -neg_reward.mean().item(), sync_dist=True)

        # if self.vanilla_ce:
        # Compute validation loss on ground truth also. (standard CE loss, not REINFORCE)
        trg_resp = b_hypothesis[:, :-1]
        y = self(b_premise.squeeze(1), trg_resp)
        score_mask = (trg_resp != 0).unsqueeze(-1).expand(-1, -1, self.vocab_size)

        neg_reward, attn_vec = self.generate_reward(b_premise.squeeze(1),
                                            b_premise_length, b_hypothesis,
                                            b_hypothesis_length)

        # Calculate the reward only for the all samples
        # Since we will incorporate both type of samples now, the average reward should increase!
        # original_reward = -neg_reward

        # print(f"REWARD: {reward}\n")
        scores = y['scores'] * score_mask
        Seq2Seq_loss = -torch.gather(scores, -1, b_hypothesis[:, 1:].unsqueeze(-1)).squeeze(-1)
        Seq2Seq_loss = Seq2Seq_loss / b_hypothesis_length.unsqueeze(1).float()
        Seq2Seq_loss = Seq2Seq_loss.sum(-1).mean(0)
        self.log("valid/ce_loss", Seq2Seq_loss.item(), sync_dist=True)

        return {
            'valid/r3': -neg_reward,
            'gen_resp': gen_resp,
            'gen_resp_length': gen_resp_length
        }

    def test_step(self, batch, batch_idx):
        b_hypothesis = batch['hypothesis']
        b_premise = batch['premise']
        b_hypothesis_length = batch['hypothesis_length']
        b_premise_length = batch['premise_length']
        b_label = batch['label']

        current_batch_size = b_hypothesis.shape[0]

        if self.enc_weight_init == "blender" or self.dec_weight_init == "blender":
            hyp = self.hf_blenderbot.generate(
                b_premise.squeeze(1),
                do_sample=True, 
                max_length=30,
                top_p=0.8, 
                top_k=0
            )
            trg_lengths_tensor = torch.argmax((hyp == self.eos_idx).long(), dim=1)
            # trg_lengths = [1+l for l in trg_lengths_tensor.tolist()] # since argmax is 0 indexed
            trg_lengths = (1 + trg_lengths_tensor) # since argmax is 0 indexed
            y = {
              'sequence': hyp,
              'sequence_len': trg_lengths.long()
            }
        else:
            y = self(b_premise.squeeze(1), max_decode_len=30, epsilon=-1, top_p=0.8, top_k=0)     

        gen_resp = y['sequence']
        gen_resp_length = y['sequence_len']
        padding = (gen_resp_length[:, None] < torch.arange(0, gen_resp.shape[1], 1)[None, :].to(gen_resp_length.device))
        gen_resp[padding] = self.pad_idx
        
        # TODO: pad the sampled sequence based on the length of the generated sequence
        # The new, optimized Seq2Seq_generate_valid function doesn't check for padding beyond EOS_ID
        neg_reward, attn_vec = self.generate_reward(b_premise,
                                                b_premise_length, gen_resp, gen_resp_length)

        self.log("test/r3", -neg_reward.mean().item(), sync_dist=True)

        return {
            'test/r3': -neg_reward,
            'gen_resp': gen_resp,
            'gen_resp_length': gen_resp_length
        }

    def configure_optimizers(self):
        # TODO: Configurable Learning Rate
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        num_training_steps = self.num_training_steps
        num_warmup_steps = 1000
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, num_warmup_steps, num_training_steps)

        lr_schedulers = [
            {
                "scheduler": scheduler, 
                "interval": "step",
                "frequency": 1,
                # "monitor": "valid/loss"
            }
        ]
        return [optimizer], lr_schedulers
