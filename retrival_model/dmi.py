import os
import math
import numpy as np
import torch
from torch import nn as nn
from transformers import RobertaModel, AutoModel, AutoTokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len= 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)#size = (max_len, 1)

        if d_model%2==0:
            div_term = torch.exp(torch.arange(0, d_model,2).float() * (-math.log(10000.0)/d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position*div_term)
        else:
            div_term = torch.exp(torch.arange(0, d_model+1, 2).float() * (-math.log(10000.0)/d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(1) # size - (max_len, 1, d_model)
        self.register_buffer('pe', pe)
        # print('POS ENC. :', pe.size()) # 5000,1,embed_size

    def forward(self, x): # 1760xbsxembed
        x = x+self.pe[:x.size(0), :, :].repeat(1, x.size(1), 1)
        return self.dropout(x)


class Embedding(nn.Module):
    def __init__(self, vocab_size=9000, d_model=512):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, ids):
        return self.emb(ids)


class Projection(nn.Module):
    def __init__(self, input_size, proj_size, dropout=0.1):
        super(Projection, self).__init__()
        self.W = nn.Linear(input_size, input_size)

    def forward(self, x):
        return self.W(x)


class Transformer(nn.Module):
    def __init__(self, d_model=512, vocab_size=9000, num_layers=2, heads=4, dim_feedforward=2048):
        super(Transformer, self).__init__()
        # self.output_len = output_len
        # self.input_len = input_len
        self.d_model = d_model
        # self.vocab_size = d_model
        self.vocab_size = vocab_size
        self.pos_encoder = PositionalEncoding(d_model)

#         self.encoder_layer = RZTXEncoderLayer(d_model=self.d_model, nhead=heads)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=heads, dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # self.emb = nn.Embedding(self.vocab_size, self.d_model)

        # self.probHead = nn.Linear(self.d_model, self.output_len)

    def forward(self, x, mask):
        # input - (batch, seq, hidden)
        # x = torch.randn(bs, input_len, d_model).to(device)
        # x = self.emb(x)
        # bs = x.size()[0]
        x = x.permute(1, 0, 2)
        pos_x = self.pos_encoder(x)
        # print(mask)
        #         print(pos_x.shape)
        encoder_output = self.encoder(pos_x, src_key_padding_mask=mask)  # (input_len, bs, d_model)
        # encoder_output = self.probHead(encoder_output)
        encoder_output = encoder_output.permute(1, 0, 2)  # torch.transpose(encoder_output, 0, 1)
        return encoder_output


class WrappedSMI(nn.Module):
    def __init__(self, model):
        super(WrappedSMI, self).__init__()
        self.module = model

    def forward(self, x):
        return self.module(x)


def identity(x):
    return x


def recursive_init(module):
    try:
        for c in module.children():
            if hasattr(c, "reset_parameters"):
                print("Reset:", c)
                c.reset_parameters()
            else:
                recursive_init(c)
    except Exception as e:
        print(module)
        print(e)


class SMI(nn.Module):
    def __init__(self, vocab_size=9000, d_model=512, projection_size=512, encoder_layers=4, encoder_heads=4,
                 dim_feedforward=2048, symmetric_loss=False, roberta_init=False, roberta_name="roberta-base"):
        super(SMI, self).__init__()
        self.d_model = d_model
        """
        If invert_mask == True,
        Then Model needs following masking protocol
        - 1 for tokens that are not masked,
        - 0 for tokens that are masked.
        """
        self.invert_mask = False
        self.roberta_init = roberta_init
        if not roberta_init:
            self.encoder = Transformer(d_model, vocab_size, encoder_layers, encoder_heads, dim_feedforward=dim_feedforward)
            self.embedding = Embedding(vocab_size, d_model)
        else:
            # ROBERTA INITIALIZE!
            self.invert_mask = True
            self.embedding = identity
            self.encoder = AutoModel.from_pretrained(roberta_name, add_pooling_layer=False)
            # self.encoder.encoder.layer[11]
            recursive_init(self.encoder.encoder.layer[-1])
        self.proj = Projection(d_model, projection_size)
        if symmetric_loss:
            self.lsoftmax0 = nn.LogSoftmax(dim=0)
            self.lsoftmax1 = nn.LogSoftmax(dim=1)
        else:
            self.lsoftmax1 = nn.LogSoftmax()
        self.symmetric_loss = symmetric_loss

        # self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for n, p in self.named_parameters():
            if p.dim() > 1:
                # print(n)
                torch.nn.init.xavier_normal_(p)

    def forward_context_only(self, context, mask_ctx, return_attn=False):
        if self.invert_mask:
            mask_ctx = (mask_ctx == 0) * 1
        context_enc = self.embedding(context)

        c_out = self.encoder(context_enc, mask_ctx, output_attentions=return_attn)
        if self.roberta_init:
            c_t = c_out.last_hidden_state[:, 0, :].contiguous()  # torch.mean(c_t, dim=1) #(batch, d)
        else:
            c_t = c_t[:, 0, :]  # torch.mean(c_t, dim=1) #(batch, d)

        if return_attn:
            head_avg_attn = torch.mean(c_out.attentions[-1],dim=1)
            attn = head_avg_attn[:, 0, :] # CLS token attention
            return c_t, attn

        return c_t

    def forward_response_with_projection(self, response, mask_resp, return_attn=False):
        if return_attn:
            r_t, attn = self.forward_context_only(response, mask_resp, return_attn)
        else:
            r_t = self.forward_context_only(response, mask_resp)

        z_t = self.proj(r_t)
        return z_t, attn

    def forward(self, context, response, mask_ctx, mask_rsp):
        if self.invert_mask:
            mask_ctx = (mask_ctx == 0) * 1
            mask_rsp = (mask_rsp == 0) * 1
        context_enc = self.embedding(context)
        response_enc = self.embedding(response)

        c_t = self.encoder(context_enc, mask_ctx)
        if self.roberta_init:
            c_t = c_t.last_hidden_state[:, 0, :].contiguous()  # torch.mean(c_t, dim=1) #(batch, d)
        else:
            c_t = c_t[:, 0, :]  # torch.mean(c_t, dim=1) #(batch, d)

        r_t = self.encoder(response_enc, mask_rsp)
        if self.roberta_init:
            z_t = r_t.last_hidden_state[:, 0, :].contiguous()  # torch.mean(z_t, dim=1) # (batch, d)
        else:
            z_t = r_t[:, 0, :]  # torch.mean(z_t, dim=1) # (batch, d)
        z_t = self.proj(z_t)  # (batch, d)

        return c_t, z_t

    @staticmethod
    def from_pretrained(checkpoint_path, roberta_init=False, roberta_name=""):
        assert os.path.exists(checkpoint_path), f"checkpoint path given to SMI.from_pretrained doesn't exist: {checkpoint_path}"
        if not roberta_init:
            raise NotImplementedError("@from_pretrained: not implemented yet for non-roberta init")
        else:
            # This will work for both roberta or bert-6L!
            print(f"[TOKENIZER] Override tokenizer as args.roberta_init is enabled: {roberta_name}")
            tokenizer = AutoTokenizer.from_pretrained(roberta_name)

        cpu = torch.device("cpu") # Load to cpu first
        checkpoint = torch.load(checkpoint_path, map_location=cpu)
        args = checkpoint['args']
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        auc = checkpoint['auc']
        state_dict = checkpoint['model_state_dict']

        cpc = SMI(
            vocab_size=len(tokenizer),
            d_model=args['d_model'],
            projection_size=args['projection'],
            encoder_layers=args['encoder_layers'],
            encoder_heads=args['encoder_heads'],
            dim_feedforward=args.get('dim_feedforward', 2048), # 2048 is the default
            roberta_init=roberta_init,
            roberta_name=roberta_name
        )
        if is_ddp_module(state_dict):
            print("*** WARNING: Model was saved as ddp module. Extracting self.module...")
            wsmi = WrappedSMI(cpc)
            load_status = wsmi.load_state_dict(state_dict, strict=False)
            cpc = wsmi.module
        else:
            load_status = cpc.load_state_dict(state_dict, strict=False)
        missing, unexpected = load_status
        if len(unexpected) > 0:
            print(
                f"\n[WARNING] ***Some weights of the model checkpoint at were not used when "
                f"initializing: {unexpected}\n"
            )
        else:
            print(f"\n[INFO] ***All model checkpoint weights were used when initializing current model.\n")
        if len(missing) > 0:
            print(
                f"\n[WARNING] ***Some weights of current model were not initialized from the model checkpoint file "
                f"and are newly initialized: {missing}\n"
            )
        assert (len(missing) <= 2 and len(unexpected) <= 2), f"Too many missing/unexpected keys in checkpoint!!!"

        print(f"Loaded pretrained model\n\tEpoch: {epoch}\n\tLoss: {loss}\n\tAUC: {auc}")

        return cpc

def is_ddp_module(state_dict):
    return 'module' in list(state_dict.keys())[0]