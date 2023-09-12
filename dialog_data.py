import json
import os
import random
import re

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import ChainDataset, DataLoader, ConcatDataset, IterableDataset, Sampler, DistributedSampler
import glob

import logging
import coloredlogs
from transformers import AutoTokenizer, BlenderbotTokenizer, BlenderbotTokenizerFast, \
    BertTokenizer, BertTokenizerFast, \
    RobertaTokenizer, RobertaTokenizerFast, \
    GPT2TokenizerFast, T5TokenizerFast, BartTokenizerFast

from r1m_preprocess import filter_dialogs
from callbacks import RandomNegativeSampler, FaissResponseSampler, NucleusResponseSampler
C_MAX_LEN = 300
R_MAX_LEN = 60

###############
# LOGGING SETUP
###############
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


class TRLWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset: BaseDataClass = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # return review, query and input_ids instead
        dx = self.dataset[idx]
        context = dx['premise']
        response = dx['hypothesis']
        context_txt = self.dataset.tokenizer.decode(context)
        response_txt = self.dataset.tokenizer.decode(response)
        return {
            "query": context_txt,
            "query_tensor": context,
            # "review": response_txt
            "response": response_txt,
            "response_tensor": response,
            "full_dialog": f"{context_txt} {self.dataset.tokenizer.sep_token} {response_txt}"
        }

    


class BaseDataClass(Dataset):
    def prep_tokenizer_info(self, tokenizer, max_ctx_len, max_resp_len):
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_resp_len = max_resp_len

        if isinstance(tokenizer, BlenderbotTokenizer) or isinstance(tokenizer, BlenderbotTokenizerFast):
            self.CLS = tokenizer.bos_token_id
            self.CLS_STR = tokenizer.bos_token
            self.EOU = tokenizer.sep_token
        elif isinstance(tokenizer, BertTokenizer) or isinstance(tokenizer, BertTokenizerFast) \
            or isinstance(tokenizer, RobertaTokenizerFast) or isinstance(tokenizer, RobertaTokenizer):
            # NO NEW TOKEN added because we may init the model with actual Roberta/bert weights
            self.CLS = tokenizer.cls_token_id
            self.CLS_STR = tokenizer.cls_token
            self.EOU = tokenizer.sep_token
        elif isinstance(tokenizer, GPT2TokenizerFast):
            self.CLS = tokenizer.bos_token_id
            self.CLS_STR = tokenizer.bos_token
            self.EOU = tokenizer.eos_token
        elif isinstance(tokenizer, T5TokenizerFast):
            tokenizer.add_special_tokens({
                'cls_token': tokenizer.eos_token,
                'sep_token': tokenizer.eos_token
            })
            self.CLS = tokenizer.eos_token_id
            self.CLS_STR = tokenizer.eos_token
            self.EOU = tokenizer.eos_token
        elif isinstance(tokenizer, BartTokenizerFast):
            self.CLS = tokenizer.cls_token_id
            self.CLS_STR = tokenizer.cls_token
            self.EOU = tokenizer.sep_token
        else:
            logger.error("Tokenizer not supported.")
            raise NotImplementedError(f"Tokenizer {tokenizer} is not supported.")

        self.pad_token_id = tokenizer.pad_token_id

        assert self.CLS is not None, "CLS token not found."
        assert self.EOU is not None, "EOU token not found."

        logger.debug(f"[TOKENIZER] cls: {self.CLS}, cls_str: {self.CLS_STR}, sep: {self.EOU}, pad: {self.pad_token_id}")
        logger.debug(f"[TOKENIZER] {tokenizer}")

    def collate_fn(self, batch):
        morphed_batch = pd.DataFrame(batch).to_dict(orient="list")
        final_batch = {
            "premise": pad_sequence(morphed_batch["premise"], batch_first=True, padding_value=self.pad_token_id),
            "hypothesis": pad_sequence(morphed_batch["hypothesis"], batch_first=True, padding_value=self.pad_token_id),
            "premise_length": torch.tensor(morphed_batch["premise_length"]),
            "hypothesis_length": torch.tensor(morphed_batch["hypothesis_length"]),
            "label": torch.tensor(morphed_batch["label"]),
            "index": torch.tensor(morphed_batch["index"])
        }
        return final_batch

    def _preprocess(self, C, R=None):
        # should be on cpu to support multiple workers in dataloader
        # for blender
        # c = self.tokenizer.encode("<s> " + C)
        # r = self.tokenizer.encode("<s> " + R)

        # for bert
        # c = self.tokenizer.encode(C)
        # r = self.tokenizer.encode(R)

        c = self.tokenizer.encode(C, add_special_tokens=False)
        l1 = len(c)
        if l1 >= self.max_ctx_len:
            c = c[l1 - self.max_ctx_len + 1:]
        c = [self.CLS] + c
        c = torch.tensor(c)

        if R is not None:
            r = self.tokenizer.encode(R, add_special_tokens=False)
            l2 = len(r)
            if l2 >= self.max_resp_len:
                r = r[:self.max_resp_len - 1]
            r = [self.CLS] + r
            r = torch.tensor(r)
        else:
            r = None

        return c, r
    
    def __len__(self):
        return len(self.data)


class DialogData(BaseDataClass):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, data_path, tokenizer, prob_positive=0.8,
                 max_ctx_len=C_MAX_LEN, max_resp_len=R_MAX_LEN, min_dial_len=2, resp_sampler_callback=None, neg_per_positive=None):
        """
        @param tokenizer: A huggingface tokenizer
        @param neg_per_positive: (npp) can be between 0 to 1 or any integer greater than 1.
        """
        super(DialogData, self).__init__()
        _file = data_path

        logger.debug(f"File: {_file}")

        self.prep_tokenizer_info(tokenizer, max_ctx_len, max_resp_len)

        # npp
        assert prob_positive > 0, "Probability of positive dialogs must be greater than 0." 
        self.prob_positive = prob_positive
        logger.debug(f"Probability of positive sample: {self.prob_positive}")

        self.dial_data = []

        with open(_file) as f:
            for line in tqdm(f, desc="Loading data"):
                # if len(self.data) > max_items:
                #     break  # Enough for now
                Full_D = line.strip().strip("__eou__").split(" __eou__ ")
                self.dial_data.append(Full_D)
        
        # need this to match the testset with other libraries!
        self.min_dial_len = min_dial_len
        self.num_positives = -1
        self.extract_cr_pairs()

        self.setup_sampler(resp_sampler_callback)

    def setup_sampler(self, resp_sampler_callback):
        # Negative/Candidate Response Sampler
        self.sampler_name = "random"
        if resp_sampler_callback is None:
            self.resp_sampler_callback = RandomNegativeSampler(len(self.data_only_positives))
        else:
            if isinstance(resp_sampler_callback, FaissResponseSampler):
                self.sampler_name = "faiss"
            elif isinstance(resp_sampler_callback, NucleusResponseSampler):
                self.sampler_name = "nucleus"
            self.resp_sampler_callback = resp_sampler_callback
        
        logger.debug(f"Changing response sampler: {self.sampler_name}")

    def extract_cr_pairs(self):
        self.data = []
        self.data_only_positives = []
        MIN_DIAL_LEN = self.min_dial_len

        for Full_D in tqdm(self.dial_data, desc="Unrolling dialogs"):
            if len(Full_D) >= MIN_DIAL_LEN:
                for j in range(MIN_DIAL_LEN, len(Full_D) + 1):
                    D = Full_D[:j]
                    C = f" {self.EOU} ".join(D[:-1]).strip() + f" {self.EOU}"
                    R = D[-1].strip() + f" {self.EOU}"
                    # mid = len(D)//2
                    # C = " __eou__ ".join(D[:mid])
                    # R = " __eou__ ".join(D[mid:])

                    # For 1 item in wo_negatives
                    self.data_only_positives.append([C, R])
                    pos_item_index = len(self.data_only_positives) - 1
                    self.data.append([C, R, pos_item_index])

        self.num_positives = len(self.data_only_positives)
        logger.debug(f"Loaded {len(self.data_only_positives)} (+) CR-samples.")
        logger.debug(f"Generated {len(self.data)} (+/-) CR-samples.")
        logger.debug(f"Samples: {self.data[random.randint(0, len(self.data))]}")

    def __getitem__(self, index):
        C, R, pos_item_index = self.data[index]
        
        # positive sample
        c, r = self._preprocess(C, R)
        label = 1
        
        if self.sampler_name in ["random", "faiss"] and random.random() > self.prob_positive:
            # negative sample
            candidate = self.resp_sampler_callback.sample(pos_item_index)
            if candidate["ni"] > -1:
                ni = candidate["ni"]
                _, R_neg = self.data_only_positives[ni]
                c, r = self._preprocess(C, R_neg)
            else:
                r = torch.tensor(candidate["cr"])
                c, _ = self._preprocess(C)

            label = 0

        return {
            "premise": c,
            "hypothesis": r,
            "premise_length": len(c),
            "hypothesis_length": len(r),
            "label": label,
            "index": pos_item_index
        }

class SimpleDSTC7Data(DialogData):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, json_path, tokenizer, prob_positive,
                 max_ctx_len=C_MAX_LEN, max_resp_len=R_MAX_LEN, min_dial_len=2, resp_sampler_callback=None):
        """
        @param tokenizer: A huggingface tokenizer
        @param neg_per_positive: (npp) can be between 0 to 1 or any integer greater than 1.
        """
        super(DialogData, self).__init__()
        _file = json_path

        logger.debug(f"File: {_file}")

        self.prep_tokenizer_info(tokenizer, max_ctx_len, max_resp_len)

        # npp
        assert prob_positive > 0, "Probability of positive dialogs must be greater than 0." 
        self.prob_positive = prob_positive
        logger.debug(f"Probability of positive sample: {self.prob_positive}")

        self.dial_data = []
                
        with open(_file) as f:
            raw_data = json.load(f)

            for i, full_log in enumerate(tqdm(raw_data, desc="Loading data")):
                messages = full_log['messages-so-far']
                dialog = []
                last_sp = "who"
                for turn in messages:
                    if turn["speaker"] != last_sp:
                        last_sp = turn["speaker"]
                        dialog.append(turn["utterance"])
                    else:
                        print(i)
                        dialog[-1] += f" {turn['utterance']}"
                if 'options-for-correct-answers' in full_log:
                    dialog.append(full_log['options-for-correct-answers'][0]['utterance'])
                self.dial_data.append(dialog)
            del raw_data

        # Filter URL and weird pieces of text (Do this before extracting CR pairs)
        _, self.dial_data = filter_dialogs(self.dial_data)
        
        # need this to match the testset with other libraries!
        self.min_dial_len = min_dial_len
        self.extract_cr_pairs()
        self.setup_sampler(resp_sampler_callback)


def extract_from_json(dev, true_responses=None):
    dev_stuff = []
    for item in tqdm(dev, desc="Extracting dialog data from .json"):
        dialog = item["messages-so-far"]
        con = []
        last_p = "none"

        # Context
        #########
        for turn in dialog:
            if turn["speaker"] == last_p:
                con[-1] = con[-1] + turn["utterance"]
            else:
                con.append(turn["utterance"])
                last_p = turn["speaker"]

        # Negative Samples + Positive/True candidate
        ############################################
        negative_samples = {}
        for option in item["options-for-next"]:
            negative_samples[option['candidate-id']] = option['utterance']

        # Get the true response
        ########################################
        if "options-for-correct-answers" in item:
            # Can't handle more than 1 true response
            # assert len(item["options-for-correct-answers"]) == 1
            gt = item["options-for-correct-answers"][0]["utterance"]
            gt_hash = item["options-for-correct-answers"][0]["candidate-id"]
        else:
            gt = true_responses[item["example-id"]]['text']
            gt_hash = true_responses[item["example-id"]]['hash']

        # Remove true candidate
        del negative_samples[gt_hash]
        # assert(len(negative_samples)) == 99
        dev_stuff.append((item["example-id"], con, gt, list(negative_samples.values())))
    # logger.debug(item)
    return dev_stuff



class SimpleRedditData(DialogData):
    """
    Similar to the DialogData class, but with a different .json data format.
    """
    def __init__(self, json_path, tokenizer, prob_positive, start=0, end=-1,
                 max_ctx_len=C_MAX_LEN, max_resp_len=R_MAX_LEN, min_dial_len=2, resp_sampler_callback=None):
        """
        @param tokenizer: A huggingface tokenizer
        @param neg_per_positive: (npp) can be between 0 to 1 or any integer greater than 1.
        @param start: (int) start index of the data
        @param end: (int) end index of the data
            start and end would be used by IterableDataset to create slices of the data.
        """
        super(DialogData, self).__init__()
        _file = json_path

        logger.debug(f"Reading file: {_file}")

        self.prep_tokenizer_info(tokenizer, max_ctx_len, max_resp_len)

        # npp
        assert prob_positive > 0, "Probability of positive dialogs must be greater than 0." 
        self.prob_positive = prob_positive
        logger.debug(f"Probability of positive sample: {self.prob_positive}")

        self.dial_data = []

        with open(_file) as f:
            for line in tqdm(f, desc="Loading data"):
                # if len(self.data) > max_items:
                #     break  # Enough for now
                Full_D = json.loads(line)["turns"]
                self.dial_data.append(Full_D)
        
        logger.debug(f"Number of dialogs in the file: {len(self.dial_data)}")
        
        # need this to match the testset with other libraries!
        self.min_dial_len = min_dial_len
        self.num_positives = -1
        self.extract_cr_pairs()
                
        logger.debug(f"Cutting data to {start}:{end}")
        self.data = self.data[start:end]
        if len(self.data) == 0:
            logger.warning(f"No data found in the given range [{start}, {end}]")

        self.setup_sampler(resp_sampler_callback)


class SequentialRedditData(Dataset):
    """
    A dataset that iterates over a list of .json reddit files.
    """
    def __init__(self, dstc8_reddit_directory, split, tokenizer, prob_positive, steps_per_epoch,
                 max_ctx_len=C_MAX_LEN, max_resp_len=R_MAX_LEN, min_dial_len=2, resp_sampler_callback=None):
        super().__init__()

        search_path = dstc8_reddit_directory + f"/{split}*.json"
        self.reddit_file_list = sorted(glob.glob(search_path))
        assert len(self.reddit_file_list) > 0, "No reddit files found in the given directory."
        logger.debug(f"Found {len(self.reddit_file_list)} reddit files in {search_path}")

        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_resp_len = max_resp_len
        self.min_dial_len = min_dial_len
        self.prob_positive = prob_positive
        self.steps_per_epoch = steps_per_epoch
        self.resp_sampler_callback = resp_sampler_callback

        self.current_file_index = 0
        self.next_start = 0

        self.pad_token_id = self.tokenizer.pad_token_id
        self.data_source = None
    
    def setup_sampler(self, resp_sampler_callback):
        self.resp_sampler_callback = resp_sampler_callback
    
    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        return self.data_source[idx]

    def _load_extra_data(self, required_items):
        """
        Loads more data from the next file.
        May not return the same number of items as asked for.
        """
        data = SimpleRedditData(self.reddit_file_list[self.current_file_index], self.tokenizer, self.prob_positive,
                            self.next_start, self.next_start + required_items, self.max_ctx_len, self.max_resp_len,
                            self.min_dial_len, self.resp_sampler_callback)
        self.next_start += required_items

        logger.debug(f"(Iterator) Added to next epoch: {len(data)} CR pairs @ {self.reddit_file_list[self.current_file_index]}")

        # check if we're done with this file
        if len(data) < required_items:
            # We're done with this file, so move on to the next one
            # If no files are left, restart from the beginning
            self.current_file_index = (self.current_file_index + 1) % len(self.reddit_file_list)
            self.next_start = 0
            if self.current_file_index == 0:
                logger.warning("No more files left. Restarting from the beginning.")
            logger.debug(f"(Iterator) Next file: {self.reddit_file_list[self.current_file_index]}")
        
        return data
        
    def ping(self):
        logger.debug("PING recieved. Updating data.")

        required_items = self.steps_per_epoch
        chain_input = []
        while required_items > 0:
            data = self._load_extra_data(required_items)
            if len(data) > 0:
                required_items -= len(data)
                chain_input.append(data)
            elif len(data) == 0:
                logger.error("No data found in the given range [{}, {}] in file {}".format(self.next_start, self.next_start + required_items, self.reddit_file_list[self.current_file_index]))
                raise Exception("Something went wrong in Reddit-IterableDataset. No data found.")

        assert required_items == 0, "Something went wrong in Reddit-IterableDataset. required_items is not 0."
        # Just for sanity check.
        # if required_items < 0:
        #     logger.error("(@iterator: required_items < 0) Something went wrong with the data loading.")
        #     raise AssertionError("Something went wrong with the data loading.")

        # if required_items > 0:
        #     # We've run out of data
        #     logger.error(f"Not enough data left to fill the epoch. {required_items} items remaining.")

        # Return a pytorch chain dataset
        self.data_source = ConcatDataset(chain_input)
         
    def collate_fn(self, batch):
        morphed_batch = pd.DataFrame(batch).to_dict(orient="list")
        final_batch = {
            "premise": pad_sequence(morphed_batch["premise"], batch_first=True, padding_value=self.pad_token_id),
            "hypothesis": pad_sequence(morphed_batch["hypothesis"], batch_first=True, padding_value=self.pad_token_id),
            "premise_length": torch.tensor(morphed_batch["premise_length"]),
            "hypothesis_length": torch.tensor(morphed_batch["hypothesis_length"]),
            "label": torch.tensor(morphed_batch["label"]),
            "index": torch.tensor(morphed_batch["index"])
        }
        return final_batch

class DatasetPingSampler(Sampler):
    # Pings the dataset before every epoch!

    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        self.data_source.ping()
        return iter(range(len(self.data_source)))


class CustomDataLoader(DataLoader):
    # Pings the dataset before every epoch!
    def __iter__(self):
        # Ping before workers get created! 
        # Hopefully this works automatically 
        # with Distributed Sampler used by Lightning
        self.dataset.ping()
        return super().__iter__()


class DSTC7Data(BaseDataClass):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, json_path, tokenizer, tsv_path=None, neg_per_positive=10.0,
                 max_ctx_len=C_MAX_LEN, max_resp_len=R_MAX_LEN, retrieval_eval_mode=False):
        """
        @param tsv_path: The ground truth response file for test set of dstc7
        @param tokenizer: A huggingface tokenizer
        @param neg_per_positive: (npp) can be between 0 to 1 or any integer greater than 1.
        """
        super(DSTC7Data, self).__init__()
        _file = json_path

        logger.debug(f"File: {_file}")

        self.prep_tokenizer_info(tokenizer, max_ctx_len, max_resp_len)

        # npp
        assert neg_per_positive >= 0, "Negative sampling rate must be greater than zero."
        if neg_per_positive >= 1:
            self.neg_per_positive = round(neg_per_positive)
            logger.debug(f"Negative sampling is set to {self.neg_per_positive} (greater than 1).")
        elif neg_per_positive > 0:
            self.neg_per_positive = neg_per_positive
            logger.debug(f"Negative sampling is set to {self.neg_per_positive} (Fractional Negative Sampling).")
        else:
            self.neg_per_positive = neg_per_positive
            logger.debug(f"No negative sampling..")

        self.dial_data = []

        # Ubuntu
        logger.debug(f"Reading json file: {json_path}")
        if tsv_path is None:
            with open(json_path) as f:
                obj = json.load(f)
                dev_stuff = extract_from_json(obj)
                del obj
        else:
            with open(tsv_path) as f:
                test_responses = {}
                for line in f:
                    id, hash, text = re.split("\t", line)
                    test_responses[int(id)] = {'hash': hash, 'text': text}
            with open(json_path) as f:
                obj = json.load(f)
                dev_stuff = extract_from_json(obj, test_responses)
                del obj

        self.dial_data = dev_stuff

        self.extract_cr_pairs()

        # Decide whether to randomize negative samples in test split
        self.retrieval_eval_mode = retrieval_eval_mode
        if self.retrieval_eval_mode:
            assert len(self.data_only_negatives[0]) == self.neg_per_positive, \
                "For retrieval_mode, first set [npp = len(negative_samples_set)]"
            logger.warning("Switching to fixed order negative sampling.")

    def extract_cr_pairs(self):
        self.data = []
        self.data_only_positives = []
        self.data_only_negatives = []

        # How often does negative samples occur, for fractional npp
        if 0 < self.neg_per_positive < 1:
            negative_sample_interval = round(1/self.neg_per_positive)
            assert negative_sample_interval >= 1, "Something wrong with ``negative_sample_interval''."

        for Full_D in tqdm(self.dial_data, desc="Constructing CR pairs"):
            _, context, gt, negatives = Full_D
            C = f" {self.EOU} ".join(context).strip() + f" {self.EOU}"
            R = gt.strip() + f" {self.EOU}"
            # mid = len(D)//2
            # C = " __eou__ ".join(D[:mid])
            # R = " __eou__ ".join(D[mid:])

            # For 1 item in wo_negatives
            self.data_only_positives.append([C, R])
            self.data_only_negatives.append(negatives)
            # there are (1+neg_per_pos) items in self.data
            self.data.append([C, R])
            if self.neg_per_positive >= 1:
                self.data.extend([[C, None]]*self.neg_per_positive)
            elif self.neg_per_positive > 0:
                # As we have less negative samples with fractional npp, we need to randomize the context also.
                if len(self.data) % negative_sample_interval == 0:
                    self.data.append([None, None])

        logger.debug(f"Loaded {len(self.data_only_positives)} (+) CR-samples.")
        logger.debug(f"Generated {len(self.data)} (+/-) CR-samples.")
        logger.debug(f"Samples: {self.data[random.randint(0, len(self.data))]}")

    def __getitem__(self, index):
        C, R = self.data[index]
        # Negative sampling >= 1
        pos_item_index = index // (1 + self.neg_per_positive)
        if R is None and C is not None:
            if not self.retrieval_eval_mode:
                # We're picking the right context as pivot
                #    tested using following assertion
                assert self.data[index][0] == self.data_only_positives[pos_item_index][0]

                ni = random.randint(0, len(self.data_only_negatives[pos_item_index]) - 1)

                R_neg = self.data_only_negatives[pos_item_index][ni]
                c, r = self._preprocess(C, R_neg)
                label = 0
            else:
                ni = (index % (1 + self.neg_per_positive) - 1)
                R_neg = self.data_only_negatives[pos_item_index][ni]
                c, r = self._preprocess(C, R_neg)
                label = 0
        elif C is None:
            index_a = random.randint(0, len(self.data_only_positives) - 1)
            C = self.data_only_positives[index_a][0]
            ni = random.randint(0, len(self.data_only_negatives[index_a]) - 1)
            R_neg = self.data_only_negatives[index_a][ni]
            c, r = self._preprocess(C, R_neg)
            label = 0
        else:
            c, r = self._preprocess(C, R)
            label = 1

        return {
            "premise": c,
            "hypothesis": r,
            "premise_length": len(c),
            "hypothesis_length": len(r),
            "label": label,
            "index": pos_item_index
        }


def validate_batch_tokens(batch, tokenizer):
    # confirm that all token ids are within range
    assert batch["premise"].min() >= 0
    assert batch["premise"].max() < tokenizer.vocab_size
    assert batch["hypothesis"].min() >= 0
    assert batch["hypothesis"].max() < tokenizer.vocab_size

if __name__=="__main__":
    logger.debug("Running Unit Tests")

    # vocab_model_reference = 'facebook/blenderbot-3B'
    vocab_model_reference = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(vocab_model_reference, use_fast=True, verbose=False)

    # Unit tests - DialogData
    split = "train"
    train_dataset = DialogData(os.path.join("/content/data", f'ijcnlp_dailydialog_cc/{split}/dialogues_{split}.txt'),
                               tokenizer, neg_per_positive=10)
    print(type(train_dataset))
    print(train_dataset.__getitem__(10))
    print(f"Length of train set: {len(train_dataset)}")
    # Speed test loop
    for i in tqdm(range(len(train_dataset)), desc="Speed test dd-train"):
        batch = train_dataset[i]
        validate_batch_tokens(batch, tokenizer)

    
    # split = "validation"
    # valid_dataset = DialogData(os.path.join("./data", f'ijcnlp_dailydialog/{split}/dialogues_{split}.txt'),
    #                            tokenizer, neg_per_positive=0.1)
    # print(f"Length of valid set: {len(valid_dataset)}")
    #
    # split = "test"
    # test_dataset = DialogData(os.path.join("./data", f'ijcnlp_dailydialog/{split}/dialogues_{split}.txt'),
    #                           tokenizer, neg_per_positive=1)
    # print(f"Length of test set: {len(test_dataset)}")
    #
    # split = "test"
    # test_dataset = DialogData(os.path.join("./data", f'ijcnlp_dailydialog/{split}/dialogues_{split}.txt'),
    #                           tokenizer, neg_per_positive=0)
    # print(f"Length of positive_only test set: {len(test_dataset)}")

    # Unit tests: DSTC7-Ubuntu
    # dev_data = DSTC7Data("data/dstc7-ubuntu/ubuntu_dev_subtask_1.json", tokenizer, neg_per_positive=10)
    # for i in tqdm(dev_data, desc="Speed test loop 1"):
    #     validate_batch_tokens(i, tokenizer)

    # test_data = DSTC7Data("data/dstc7-ubuntu/ubuntu_test_subtask_1.json", tokenizer,
    #                       tsv_path="data/dstc7-ubuntu/ubuntu_responses_subtask_1.tsv",
    #                       neg_per_positive=99, retrieval_eval_mode=True)
    # for i in tqdm(test_data, desc="Speed test loop 2"):
    #     validate_batch_tokens(i, tokenizer)
    '''
    train_data = DSTC7Data("data/dstc7-ubuntu/ubuntu_train_subtask_1.json", tokenizer, neg_per_positive=10)
    for i in tqdm(train_data, desc="Speed test loop 3"):
        try:
            validate_batch_tokens(i, tokenizer)
        except AssertionError as e:
            # Decode and print batch and find out the token that caused the error
            print(f"Error in batch: {i}")
            print(f"Vocab Range: {tokenizer.vocab_size}")
            print(f"Vocab Len: {len(tokenizer)}")
            logger.debug(f"Premise: {tokenizer.decode(i['premise'])}")
            logger.debug(f"Hyp: {tokenizer.decode(i['hypothesis'])}")
            raise e
    '''