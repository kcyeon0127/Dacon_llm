from datasets import Dataset
import numpy as np
import pandas as pd
import torch,random,json
from transformers.data import DataCollatorForSeq2Seq
from typing import Tuple,List,Dict


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths)
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)


def get_dataset(tokenizer,split:str):
    train_path = "/home/elicer/DaconAcc/dataset/unique_train_drop_prompt_0.85.csv" #"/home/elicer/DaconAcc/dataset/train_drop_prompt.csv"
    valid_path = "/home/elicer/DaconAcc/dataset/valid_prompt.csv"
    path = train_path if split == "train" else valid_path
    
    dataset = pd.read_csv(path)
    
    dataset = Dataset.from_pandas(dataset)
    print(dataset)

    def tokenize_add_label(sample):
        question = [
            {"role": "user", "content": sample["question"]}
        ]
        generation_token = "<|start_header_id|>assistant<|end_header_id|>"
        system_token = "<|start_header_id|>system<|end_header_id|>\n\n친절한 건설안전전문가로서 상대방의 요청에 최대한 '자세하고' 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘.<|eot_id|>"
        bos_token = "<|begin_of_text|>"
        
        encoded_input = tokenizer.apply_chat_template(question, tokenize=False, add_generation_prompt=False).strip()
        if encoded_input.strip()[:len(bos_token)] == bos_token:
            encoded_input = encoded_input[len(bos_token):]
        encoded_input = tokenizer.bos_token + system_token + encoded_input
        encoded_input = tokenizer.encode(encoded_input, add_special_tokens=False)
            
        answer = [
            {"role": "assistant", "content": sample["answer"]}
        ]
        encoded_output = tokenizer.apply_chat_template(answer, tokenize=False, add_generation_prompt=False).strip()
        if encoded_output.strip()[-len(generation_token):] == generation_token:
            encoded_output = encoded_output[:-len(generation_token)]
        if encoded_output.strip()[:len(bos_token)] == bos_token:
            encoded_output = encoded_output[len(bos_token):]
        
        encoded_output = encoded_output + tokenizer.eos_token
        
        encoded_output = tokenizer.encode(encoded_output, add_special_tokens=False)
        sample = {
            "input_ids": encoded_input + encoded_output,
            "attention_mask" : [1] * (len(encoded_input) + len(encoded_output)),
            "labels": [-100] * len(encoded_input) + encoded_output,
        }

        return sample
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    print()
    print(tokenizer.decode(dataset[0]["input_ids"]))
    print()
    return dataset


def get_valid_input_dataset(tokenizer):
    path = "/home/elicer/DaconAcc/dataset/valid_prompt.csv"
    dataset = pd.read_csv(path)
    
    dataset = Dataset.from_pandas(dataset)

    def tokenize_add_label(sample):
        question = [
            {"role": "user", "content": sample["question"]}
        ]
        system_token = "<|start_header_id|>system<|end_header_id|>\n\n친절한 건설안전전문가로서 상대방의 요청에 최대한 '자세하고' 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘.<|eot_id|>"
        bos_token = "<|begin_of_text|>"
        
        encoded_input = tokenizer.apply_chat_template(question, tokenize=False, add_generation_prompt=True).strip()
        if encoded_input.strip()[:len(bos_token)] == bos_token:
            encoded_input = encoded_input[len(bos_token):]
        encoded_input = tokenizer.bos_token + system_token + encoded_input
        
        sample = {
            "text": encoded_input
        }

        return sample
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    print()
    print(dataset[0]["text"])
    print()
    return dataset


def get_dataloader_kwargs(train_config, dataset, tokenizer, mode):
        kwargs = {}
        batch_size = train_config.batch_size_training if mode=="train" else train_config.val_batch_size
        kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=mode=="train")
        kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
        return kwargs