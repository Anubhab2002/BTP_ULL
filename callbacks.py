import random
# import faiss
import pytorch_lightning as pl
import coloredlogs
import logging
import numpy as np
import torch
from torch import embedding

###############
# LOGGING SETUP
###############
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

class BaseResponseSampler(pl.Callback):
    def sample(self, pos_item_index):
        raise NotImplementedError("You need to implement this method in your subclass")


class RandomNegativeSampler(BaseResponseSampler):
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def sample(self, pos_item_index):
        neg_item_index = pos_item_index
        while neg_item_index == pos_item_index:
            diff = random.randint(1, self.pool_size - 1)
            neg_item_index = (pos_item_index+diff) % self.pool_size
        
        # assert pos_item_index != neg_item_index
        
        return {
            "ni": neg_item_index
        }


class FaissResponseSampler(BaseResponseSampler):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.faiss_index = None
        # self.embedding_size = embedding_size
        self.pool = [None]*pool_size
        self.current_pool_size = 0
    
    def _build_index(self):
        if self.current_pool_size < self.pool_size:
            logger.error("Not enough samples in the pool to build the index")
            return
        logger.debug("Building Faiss index")
        nlist=50
        m = 8  # number of centroid IDs in final compressed vectors
        bits = 8 # number of bits in each centroid
        embedding_size = self.pool[0].shape[0]

        quantizer = faiss.IndexFlatIP(embedding_size) 
        self.faiss_index = faiss.IndexIVFPQ(quantizer, embedding_size, nlist, m, bits)

        self.faiss_index.train(np.stack(self.pool))
        self.faiss_index.add(np.stack(self.pool))

    def sample(self, pos_item_index):
        if self.faiss_index is None:
            neg_item_index = pos_item_index
            while neg_item_index == pos_item_index:
                diff = random.randint(1, self.pool_size - 1)
                neg_item_index = (pos_item_index+diff) % self.pool_size
            
            return {
                "ni": neg_item_index
            }
        else:
            pos_context_vector = self.pool[pos_item_index][None, :]
            dists, neighbors = self.faiss_index.search(pos_context_vector, 20)
            neighbors = neighbors[0]
            neg_item_index = random.choice(neighbors)
            while neg_item_index == pos_item_index:
                neg_item_index = random.choice(neighbors)
            return {
                "ni": neg_item_index
            }

    def add(self, indices, context_embeddings):
        for i, ix in enumerate(indices):
            if self.pool[ix] is None:
                self.pool[ix] = context_embeddings[i]
                self.current_pool_size += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused = 0):
        if self.current_pool_size < self.pool_size:
            context_embeddings = pl_module.r3_model.forward_context_only(batch['premise'], batch['premise_length']).cpu().numpy()
            self.add(batch['index'].long().tolist(), context_embeddings)

    def on_epoch_end(self, trainer, pl_module):
        if self.faiss_index is None:
            self._build_index()


class NucleusResponseSampler(BaseResponseSampler):
    def __init__(self, prob_positive, tokenizer):
        super().__init__()

        self.prob_positive = prob_positive
        self.pad_idx = tokenizer.pad_token_id

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, unused = 0):
        if random.random() < self.prob_positive:
            return

        # Negative sampling with prob = (1 - prob_positive)
        # Generate top_p samples
        b_hypothesis = batch['hypothesis']
        b_premise = batch['premise']
        b_hypothesis_length = batch['hypothesis_length']
        b_premise_length = batch['premise_length']
        b_label = batch['label']

        current_batch_size = b_hypothesis.shape[0]

        # This block is to compensate for training speed
        if random.random() < 0.5:
            # epsilon = 0 is complete sampling
            # epsilon = 1 is complete greedy
            # epsilon = -1 is top_k + top_p sampling
            y = pl_module(b_premise.squeeze(1), max_decode_len=30, epsilon=-1, top_p=0.8)
        else:
            y = pl_module(b_premise.squeeze(1), max_decode_len=30, epsilon=0)

        gen_resp = y['sequence']
        gen_resp_length = y['sequence_len']
        padding = (gen_resp_length[:, None] < torch.arange(0, gen_resp.shape[1], 1)[None, :].to(gen_resp_length.device))
        gen_resp[padding] = self.pad_idx

        batch['negative_sample'] = gen_resp
        batch['negative_sample_length'] = gen_resp_length
 
        return

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_train_epoch_end(trainer, pl_module)