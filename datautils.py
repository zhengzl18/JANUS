# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random

import numpy as np
import torch
from datasets import load_dataset

"""
This file is adapted from https://github.com/huggingface/peft/blob/main/examples/corda_finetuning/datautils.py
"""


CACHE_ROOT = "your_cache_dir"  # Set this to your desired cache directory

def get_knowledge_data(name, tokenizer, model_id, nsamples, seed=3):
    print(f" get_data_from: {name}, nsamples={nsamples}, seed={seed}")
    cache_file = f"{CACHE_ROOT}/knowledge_data/{name}_{model_id.replace('/', '_')}_{nsamples}_{seed}.pt"
    knowledge_dataset = []
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f"found data file: {cache_file}")
        knowledge_dataset = torch.load(cache_file)
        print("loaded ...")
        return knowledge_dataset

    if name == "trivia_qa":
        traindata = load_dataset("trivia_qa", "rc.nocontext", split="train").shuffle(seed=seed).take(nsamples)
        PROMPT = "Question: {question}?\nAnswer:{answer}"
        input_texts = [
            PROMPT.format(question=q, answer=a[0])
            for q, a in zip(traindata["question"], traindata["answer"]["aliases"])
        ]
    elif name == "nqopen":
        traindata = load_dataset("nq_open", split="train").shuffle(seed=seed).take(nsamples)
        PROMPT = "Answer these questions:\n\nQ: {question}?\nAnswer:{answer}"
        input_texts = [
            PROMPT.format(question=q, answer=a[0])
            for q, a in zip(traindata["question"], traindata["answer"])
        ]
    elif name == "webqs":
            traindata = load_dataset("web_questions", split="train").shuffle(seed=seed).take(nsamples)
            PROMPT = "Question: {question}\nAnswer:{answer}"
            input_texts = [
                PROMPT.format(question=q, answer=a[0])
                for q, a in zip(traindata["question"], traindata["answers"])
            ]
    else:
        raise NotImplementedError

    for text in input_texts:
        enc = tokenizer(text, return_tensors="pt")
        knowledge_dataset.append(
            {
                "input_ids": enc.input_ids, 
                "attention_mask": enc.attention_mask,
                "labels": enc.input_ids,
            }
        )
    torch.save(knowledge_dataset, cache_file)
    return knowledge_dataset