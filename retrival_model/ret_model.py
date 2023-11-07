"""
Definition of the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

import logging
import coloredlogs
from .layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from .utils import get_mask, replace_masked
from .dmi import SMI

from transformers import BertModel


# LOGGING SETUP
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

class LitESIM(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes,
                 padding_idx, neg_per_positive, learning_rate=1e-3, embeddings_init=None, dropout=0.5,
                 dataset="dd",
                 retrieval_model_type="ESIM",
                 batch_size=64):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.learning_rate = learning_rate
        
        self.neg_per_positive = neg_per_positive
        self.dataset = dataset
        self.retrieval_model_type = retrieval_model_type
        self.batch_size = batch_size

        self.save_hyperparameters()

        # Metrics
        self.train_f1 = torchmetrics.F1Score()
        self.valid_f1 = torchmetrics.F1Score()
        self.test_f1 = torchmetrics.F1Score()

        self.valid_recall_1 = torchmetrics.RetrievalRecall(k=1)
        self.test_recall_1 = torchmetrics.RetrievalRecall(k=1)
        self.test_recall_5 = torchmetrics.RetrievalRecall(k=5)
        self.test_recall_10 = torchmetrics.RetrievalRecall(k=10)

        # Loss Function
        # No ignore index as we are doing a simple binary classification
        class_weights = torch.tensor([1/(1+neg_per_positive), neg_per_positive/(1+neg_per_positive)])
        print("Retrieval class weights:", class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)


        self._word_embedding = nn.Embedding(self.vocab_size,
                                    self.embedding_dim,
                                    padding_idx=padding_idx,
                                    _weight=embeddings_init)

        self._rnn_dropout = RNNDropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

    def forward_context_only(self, premise, premise_length):
        """
        Inference step for the context only.
        :param premise:
        :param premise_length:
        :return:
        """
        with torch.no_grad():
            # Embed the premise
            premise_embedded = self._word_embedding(premise)
            # Apply dropout
            premise_embedded = self._rnn_dropout(premise_embedded)
            # Encode the premise
            premise_encoded = self._encoding(premise_embedded, premise_length)
            # cls token embedding
            return premise_encoded[:, 0, :]

    def forward(self, premises, premises_lengths, hypotheses, hypotheses_lengths):
        """
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        premises_mask = (premises != self.padding_idx).float() # Replaced get_mask from utils.py
        hypotheses_mask = (hypotheses != self.padding_idx).float()

        embedded_premises = self._word_embedding(premises)
        embedded_hypotheses = self._word_embedding(hypotheses)

        # if self.dropout:
        embedded_premises = self._rnn_dropout(embedded_premises)
        embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding(embedded_premises,
                                          premises_lengths)
        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_lengths)

        attended_premises, attended_hypotheses, attn_vec =\
            self._attention(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask)

        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        # if self.dropout:
        projected_premises = self._rnn_dropout(projected_premises)
        projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities, attn_vec

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=1)
        lr_schedulers = {"scheduler": scheduler, "monitor": "valid/f1_score"}
        return [optimizer], lr_schedulers

    def training_step(self, batch, batch_idx):
        premises = batch["premise"]
        premises_lengths = batch["premise_length"]
        hypotheses = batch["hypothesis"]
        hypotheses_lengths = batch["hypothesis_length"]
        labels = batch["label"]

        logits, probs, _ = self(premises,
                                 premises_lengths,
                                 hypotheses,
                                 hypotheses_lengths)
        loss = self.criterion(logits, labels)

        ## Logging
        self.train_f1(probs.argmax(-1), labels)
        self.log("train/loss", loss, on_epoch=True)
        self.log("train/f1_score", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        premises = batch["premise"]
        premises_lengths = batch["premise_length"]
        hypotheses = batch["hypothesis"]
        hypotheses_lengths = batch["hypothesis_length"]
        labels = batch["label"]
        indexes = batch["index"]

        logits, probs, _ = self(premises,
                                 premises_lengths,
                                 hypotheses,
                                 hypotheses_lengths)
        val_loss = self.criterion(logits, labels)

        ## Logging
        self.valid_f1(probs.argmax(-1), labels)
        self.valid_recall_1(probs[..., 1], labels, indexes)
        self.log("valid/loss", val_loss)
        self.log("valid/f1_score", self.valid_f1)
        self.log("valid/recall_at_1", self.valid_recall_1)
        return val_loss

    def test_step(self, batch, batch_idx):
        premises = batch["premise"]
        premises_lengths = batch["premise_length"]
        hypotheses = batch["hypothesis"]
        hypotheses_lengths = batch["hypothesis_length"]
        labels = batch["label"]
        indexes = batch["index"]

        logits, probs, _ = self(premises,
                                 premises_lengths,
                                 hypotheses,
                                 hypotheses_lengths)
        test_loss = self.criterion(logits, labels)

        ## Logging
        self.test_f1(probs.argmax(-1), labels)
        self.test_recall_1(probs[..., 1], labels, indexes)
        self.test_recall_5(probs[..., 1], labels, indexes)
        self.test_recall_10(probs[..., 1], labels, indexes)
        self.log("test/loss", test_loss)
        self.log("test/f1_score", self.test_f1)
        self.log("test/recall_at_1", self.test_recall_1)
        self.log("test/recall_at_5", self.test_recall_5)
        self.log("test/recall_at_10", self.test_recall_10)
        return test_loss


class LitBERT(pl.LightningModule):
    def __init__(self,
                 model_name_or_path,
                 num_classes,
                 dropout,
                 neg_per_positive,
                 padding_idx,
                 ff_dims=[1024, 256],
                 learning_rate=1e-3,
                 dataset="dd",
                 retrieval_model_type="BERT",
                 batch_size=64):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.num_classes = num_classes
        self.dropout = dropout
        self.neg_per_positive = neg_per_positive
        self.padding_idx = padding_idx
        self.ff_dims = ff_dims
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.retrieval_model_type = retrieval_model_type
        self.batch_size = batch_size

        assert len(self.ff_dims) >= 1, "At least one hidden layer is required"

        self.save_hyperparameters()        

        # Metrics
        self.train_f1 = torchmetrics.F1Score()
        self.valid_f1 = torchmetrics.F1Score()
        self.test_f1 = torchmetrics.F1Score()

        self.valid_recall_1 = torchmetrics.RetrievalRecall(k=1)
        self.test_recall_1 = torchmetrics.RetrievalRecall(k=1)
        self.test_recall_5 = torchmetrics.RetrievalRecall(k=5)
        self.test_recall_10 = torchmetrics.RetrievalRecall(k=10)

        # Loss Function
        # No ignore index as we are doing a simple binary classification
        class_weights = torch.tensor([1/(1+neg_per_positive), neg_per_positive/(1+neg_per_positive)])
        print("Retrieval class weights:", class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Add input layer to ff_dims, this will only work for bert models
        ext_ff_dims = [768*4] + self.ff_dims

        # Model
        self.bert = BertModel.from_pretrained(model_name_or_path)
        layers = nn.ModuleList()
        for i in range(len(ext_ff_dims)):
            layers.append(nn.Dropout(p=self.dropout))
            if i<len(ext_ff_dims)-1:
                layers.append(nn.Linear(ext_ff_dims[i],ext_ff_dims[i+1]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(ext_ff_dims[i],num_classes))
        self._classification = nn.Sequential(*layers)

    def forward_context_only(self, premise, premise_length):
        """
        Forward pass an input context/utterance through the model
        Return: cls embedding of last two layer of the bert-based core
        """
        with torch.no_grad():
            premise_mask = (premise != self.padding_idx).float()

            pout = self.bert(input_ids=premise, attention_mask=premise_mask, output_hidden_states=True,output_attentions=True, return_dict=True)
            pcls_states = torch.cat([pout.hidden_states[i][:,0] for i in [-2,-1]],dim=-1)

            return pcls_states

    def forward(self, premise, premise_length, hypothesis, hypothesis_length):
        premise_mask = (premise != self.padding_idx).float()
        hypothesis_mask = (hypothesis != self.padding_idx).float()

        pout = self.bert(input_ids=premise, attention_mask=premise_mask, output_hidden_states=True,output_attentions=True, return_dict=True)
        pcls_states = torch.cat([pout.hidden_states[i][:,0] for i in [-2,-1]],dim=-1)

        hout = self.bert(input_ids=hypothesis, attention_mask=hypothesis_mask, output_hidden_states=True,output_attentions=True, return_dict=True)
        hcls_states = torch.cat([hout.hidden_states[i][:,0] for i in [-2,-1]],dim=-1)
        
        logits = self._classification(torch.cat([pcls_states, hcls_states], dim=-1))
        probabilities = F.softmax(logits, dim=-1)
        
        # Resp attn is already normalized => a prob distribution
        head_avg_attn = torch.mean(hout.attentions[-1],dim=1)
        resp_attn = head_avg_attn[:, 0, :] 
        return logits, probabilities, resp_attn

    def configure_optimizers(self):
        # Differen LR for bert and classifier
        optimizer = torch.optim.Adam(
            [
                {'params': self.bert.parameters(), 'lr': self.learning_rate/20},
                {'params': self._classification.parameters(), 'lr': self.learning_rate}
            ]
        )
        print("Optimizer:", optimizer)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=1)
        lr_schedulers = {"scheduler": scheduler, "monitor": "valid/f1_score"}
        return [optimizer], lr_schedulers

    def training_step(self, batch, batch_idx):
        premises = batch["premise"]
        premises_lengths = batch["premise_length"]
        hypotheses = batch["hypothesis"]
        hypotheses_lengths = batch["hypothesis_length"]
        labels = batch["label"]

        logits, probs, _ = self(premises, premises_lengths, hypotheses, hypotheses_lengths)
        loss = self.criterion(logits, labels)

        ## Logging
        self.train_f1(probs.argmax(-1), labels)
        self.log("train/loss", loss, on_epoch=True)
        self.log("train/f1_score", self.train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        premises = batch["premise"]
        premises_lengths = batch["premise_length"]
        hypotheses = batch["hypothesis"]
        hypotheses_lengths = batch["hypothesis_length"]
        labels = batch["label"]
        indexes = batch["index"]

        logits, probs, _ = self(premises, premises_lengths, hypotheses, hypotheses_lengths)
        val_loss = self.criterion(logits, labels)

        ## Logging
        self.valid_f1(probs.argmax(-1), labels)
        self.valid_recall_1(probs[..., 1], labels, indexes)
        self.log("valid/loss", val_loss)
        self.log("valid/f1_score", self.valid_f1)
        self.log("valid/recall_at_1", self.valid_recall_1)
        return val_loss

    def test_step(self, batch, batch_idx):
        premises = batch["premise"]
        premises_lengths = batch["premise_length"]
        hypotheses = batch["hypothesis"]
        hypotheses_lengths = batch["hypothesis_length"]
        labels = batch["label"]
        indexes = batch["index"]

        logits, probs, _ = self(premises, premises_lengths, hypotheses, hypotheses_lengths)
        test_loss = self.criterion(logits, labels)

        ## Logging
        self.test_f1(probs.argmax(-1), labels)
        self.test_recall_1(probs[..., 1], labels, indexes)
        self.test_recall_5(probs[..., 1], labels, indexes)
        self.test_recall_10(probs[..., 1], labels, indexes)
        self.log("test/loss", test_loss)
        self.log("test/f1_score", self.test_f1)
        self.log("test/recall_at_1", self.test_recall_1)
        self.log("test/recall_at_5", self.test_recall_5)
        self.log("test/recall_at_10", self.test_recall_10)
        return test_loss


class LitDMI(pl.LightningModule):
    def __init__(self,
                 model_path,
                 num_classes,
                 dropout,
                 neg_per_positive,
                 padding_idx,
                 ff_dims=[1024, 256],
                 learning_rate=1e-3,
                 dataset="dd",
                 retrieval_model_type="DMI",
                 batch_size=64,
                 roberta_init=True,
                 roberta_name="google/bert_uncased_L-8_H-768_A-12"):
        """
        Args:
            model_path (str): path to the pretrained model
            roberta_init (bool): For DMI, whether to initialize the model with the pretrained roberta weights
            roberta_name (str): For DMI, name of the pretrained roberta model
                "google/bert_uncased_L-8_H-768_A-12" or "roberta-base"
        """
        super().__init__()

        self.model_path = model_path
        self.num_classes = num_classes
        self.dropout = dropout
        self.neg_per_positive = neg_per_positive
        self.padding_idx = padding_idx
        self.ff_dims = ff_dims
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.retrieval_model_type = retrieval_model_type
        self.batch_size = batch_size

        assert len(self.ff_dims) >= 1, "At least one hidden layer is required"

        self.save_hyperparameters()        

        # Metrics
        self.train_f1 = torchmetrics.F1Score()
        self.valid_f1 = torchmetrics.F1Score()
        self.test_f1 = torchmetrics.F1Score()

        self.valid_recall_1 = torchmetrics.RetrievalRecall(k=1)
        self.test_recall_1 = torchmetrics.RetrievalRecall(k=1)
        self.test_recall_5 = torchmetrics.RetrievalRecall(k=5)
        self.test_recall_10 = torchmetrics.RetrievalRecall(k=10)

        # Loss Function
        # No ignore index as we are doing a simple binary classification
        class_weights = torch.tensor([1/(1+neg_per_positive), neg_per_positive/(1+neg_per_positive)])
        print("Retrieval class weights:", class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)        

        # Model
        self.dmi = SMI.from_pretrained(self.model_path, roberta_init=roberta_init, roberta_name=roberta_name)        

        # Add input layer to ff_dims, this will only work for bert models
        ext_ff_dims = [self.dmi.d_model*2] + self.ff_dims
        layers = nn.ModuleList()
        for i in range(len(ext_ff_dims)):
            layers.append(nn.Dropout(p=self.dropout))
            if i<len(ext_ff_dims)-1:
                layers.append(nn.Linear(ext_ff_dims[i],ext_ff_dims[i+1]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(ext_ff_dims[i],num_classes))
        self._classification = nn.Sequential(*layers)

    def forward(self, premise, premise_length, hypothesis, hypothesis_length):
        c_t = self.dmi.forward_context_only(premise, premise == self.padding_idx)

        z_t, attn = self.dmi.forward_context_only(hypothesis, hypothesis == self.padding_idx, return_attn=True)
        c_t = torch.cat([c_t, z_t], dim=-1)

        logits = self._classification(c_t)
        probabilities = F.softmax(logits, dim=-1)
        return logits, probabilities, attn

    def configure_optimizers(self):
        # Differen LR for dmi and classifier
        optimizer = torch.optim.Adam(
            [
                {'params': self.dmi.parameters(), 'lr': self.learning_rate/20},
                {'params': self._classification.parameters(), 'lr': self.learning_rate}
            ]
        )
        print("Optimizer:", optimizer)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=1)
        lr_schedulers = {"scheduler": scheduler, "monitor": "valid/f1_score"}
        return [optimizer], lr_schedulers

    def training_step(self, batch, batch_idx):
        premises = batch["premise"]
        premises_lengths = batch["premise_length"]
        hypotheses = batch["hypothesis"]
        hypotheses_lengths = batch["hypothesis_length"]
        labels = batch["label"]

        logits, probs, _ = self(premises, premises_lengths, hypotheses, hypotheses_lengths)
        loss = self.criterion(logits, labels)

        ## Logging
        self.train_f1(probs.argmax(-1), labels)
        self.log("train/loss", loss, on_epoch=True)
        self.log("train/f1_score", self.train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        premises = batch["premise"]
        premises_lengths = batch["premise_length"]
        hypotheses = batch["hypothesis"]
        hypotheses_lengths = batch["hypothesis_length"]
        labels = batch["label"]
        indexes = batch["index"]

        logits, probs, _ = self(premises, premises_lengths, hypotheses, hypotheses_lengths)
        val_loss = self.criterion(logits, labels)

        ## Logging
        self.valid_f1(probs.argmax(-1), labels)
        self.valid_recall_1(probs[..., 1], labels, indexes)
        self.log("valid/loss", val_loss)
        self.log("valid/f1_score", self.valid_f1)
        self.log("valid/recall_at_1", self.valid_recall_1)
        return val_loss

    def test_step(self, batch, batch_idx):
        premises = batch["premise"]
        premises_lengths = batch["premise_length"]
        hypotheses = batch["hypothesis"]
        hypotheses_lengths = batch["hypothesis_length"]
        labels = batch["label"]
        indexes = batch["index"]

        logits, probs, _ = self(premises, premises_lengths, hypotheses, hypotheses_lengths)
        test_loss = self.criterion(logits, labels)

        ## Logging
        self.test_f1(probs.argmax(-1), labels)
        self.test_recall_1(probs[..., 1], labels, indexes)
        self.test_recall_5(probs[..., 1], labels, indexes)
        self.test_recall_10(probs[..., 1], labels, indexes)
        self.log("test/loss", test_loss)
        self.log("test/f1_score", self.test_f1)
        self.log("test/recall_at_1", self.test_recall_1)
        self.log("test/recall_at_5", self.test_recall_5)
        self.log("test/recall_at_10", self.test_recall_10)
        return test_loss


class LitDMIZero(pl.LightningModule):
    # Zero-shot DMI
    # An R3 model that uses the DMI model as it is, without
    # further fine-tuning.
    def __init__(self,
                 model_path,
                 neg_per_positive,
                 padding_idx,
                 retrieval_model_type="DMIZ_BASE",
                 is_symmetric=False,
                 roberta_init=True,
                 roberta_name="google/bert_uncased_L-8_H-768_A-12"):
        """
        Args:
            model_path (str): path to the pretrained model
            roberta_init (bool): For DMI, whether to initialize the model with the pretrained roberta weights
            roberta_name (str): For DMI, name of the pretrained roberta model
                "google/bert_uncased_L-8_H-768_A-12" or "roberta-base"
        """
        super().__init__()

        self.model_path = model_path
        # self.num_classes = num_classes
        # self.dropout = dropout
        self.neg_per_positive = neg_per_positive
        self.padding_idx = padding_idx
        # self.ff_dims = ff_dims
        # self.learning_rate = learning_rate
        # self.dataset = dataset
        self.retrieval_model_type = retrieval_model_type
        self.is_symmetric = is_symmetric
        # self.batch_size = batch_size

        # if self.neg_per_positive > self.batch_size:
        #     logger.warning("In case of DMIZero, it is recommended to use a batch size of at least (neg_per_positive)")

        # assert len(self.ff_dims) >= 1, "At least one hidden layer is required"

        self.save_hyperparameters()        

        # Metrics
        # self.train_f1 = torchmetrics.F1Score()
        self.valid_f1 = torchmetrics.F1Score()
        self.test_f1 = torchmetrics.F1Score()

        self.valid_recall_1 = torchmetrics.RetrievalRecall(k=1)
        self.test_recall_1 = torchmetrics.RetrievalRecall(k=1)
        self.test_recall_5 = torchmetrics.RetrievalRecall(k=5)
        self.test_recall_10 = torchmetrics.RetrievalRecall(k=10)

        # Loss Function
        # No ignore index as we are doing a simple binary classification
        class_weights = torch.tensor([1/(1+neg_per_positive), neg_per_positive/(1+neg_per_positive)])
        print("Retrieval class weights:", class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)        

        # Model
        self.dmi = SMI.from_pretrained(self.model_path, roberta_init=roberta_init, roberta_name=roberta_name)        

        # Add input layer to ff_dims, this will only work for bert models
        # ext_ff_dims = [self.dmi.d_model*2] + self.ff_dims
        # layers = nn.ModuleList()
        # for i in range(len(ext_ff_dims)):
        #     layers.append(nn.Dropout(p=self.dropout))
        #     if i<len(ext_ff_dims)-1:
        #         layers.append(nn.Linear(ext_ff_dims[i],ext_ff_dims[i+1]))
        #         layers.append(nn.ReLU())
        #     else:
        #         layers.append(nn.Linear(ext_ff_dims[i],num_classes))
        # self._classification = nn.Sequential(*layers)

    def forward(self, premise, premise_length, hypothesis, hypothesis_length):
        c_t = self.dmi.forward_context_only(premise, premise == self.padding_idx)

        z_t, attn = self.dmi.forward_response_with_projection(hypothesis, hypothesis == self.padding_idx, return_attn=True)
        # c_t = torch.cat([c_t, z_t], dim=-1)

        logits = c_t @ z_t.t()
        probabilities = torch.diag(F.softmax(logits, dim=1))
        # Stack p and (1-p)
        probs = torch.stack([1-probabilities, probabilities], dim=-1)
        # Compute logits for concated probs
        # logits = torch.log(probs)

        # DMI-sym
        if self.is_symmetric:
            p_vertical = torch.diag(F.softmax(logits, dim=0))
            probs_v = torch.stack([1-p_vertical, p_vertical], dim=-1)
            logits = 0.5*(torch.log(probs) + torch.log(probs_v))
        else:
            logits = torch.log(probs)
        return logits, probs, attn


    def training_step(self, batch, batch_idx):
        # premises = batch["premise"]
        # premises_lengths = batch["premise_length"]
        # hypotheses = batch["hypothesis"]
        # hypotheses_lengths = batch["hypothesis_length"]
        # labels = batch["label"]

        # logits, probs, _ = self(premises, premises_lengths, hypotheses, hypotheses_lengths)
        # loss = self.criterion(logits, labels)

        # ## Logging
        # self.train_f1(probs.argmax(-1), labels)
        # self.log("train/loss", loss, on_epoch=True)
        # self.log("train/f1_score", self.train_f1, on_step=False, on_epoch=True)

        # return loss
        return

    def validation_step(self, batch, batch_idx):
        premises = batch["premise"]
        premises_lengths = batch["premise_length"]
        hypotheses = batch["hypothesis"]
        hypotheses_lengths = batch["hypothesis_length"]
        labels = batch["label"]
        indexes = batch["index"]

        logits, probs, _ = self(premises, premises_lengths, hypotheses, hypotheses_lengths)
        val_loss = self.criterion(logits, labels)

        ## Logging
        self.valid_f1(probs.argmax(-1), labels)
        self.valid_recall_1(probs[..., 1], labels, indexes)
        self.log("valid/loss", val_loss)
        self.log("valid/f1_score", self.valid_f1)
        self.log("valid/recall_at_1", self.valid_recall_1)
        return val_loss

    def test_step(self, batch, batch_idx):
        premises = batch["premise"]
        premises_lengths = batch["premise_length"]
        hypotheses = batch["hypothesis"]
        hypotheses_lengths = batch["hypothesis_length"]
        labels = batch["label"]
        indexes = batch["index"]

        logits, probs, _ = self(premises, premises_lengths, hypotheses, hypotheses_lengths)
        test_loss = self.criterion(logits, labels)

        ## Logging
        self.test_f1(probs.argmax(-1), labels)
        self.test_recall_1(probs[..., 1], labels, indexes)
        self.test_recall_5(probs[..., 1], labels, indexes)
        self.test_recall_10(probs[..., 1], labels, indexes)
        self.log("test/loss", test_loss)
        self.log("test/f1_score", self.test_f1)
        self.log("test/recall_at_1", self.test_recall_1)
        self.log("test/recall_at_5", self.test_recall_5)
        self.log("test/recall_at_10", self.test_recall_10)
        return test_loss