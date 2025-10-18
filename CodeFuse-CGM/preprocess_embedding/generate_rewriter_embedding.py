"""
generate embedding for Queries from Rewriter's Inferer
"""

from transformers import AutoTokenizer, AutoModel
import torch
import os
import numpy as np
import pandas as pd
import tqdm
import json
import pickle

# custom
import argparse, logging

# input path
rewriter_output_path = "rewriter_output.json"

# save path
rewriter_embedding_path = "rewriter_embedding.pkl"

# load model
model_name_or_path = "xxx/CodeFuse-CGE-Large"
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, truncation_side='right', padding_side='right')

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
model.to(device)

if __name__ == "__main__":

    with open(rewriter_output_path, 'r') as file:
        rewriter_output_dict = json.load(file)

    query_embedding_dict = {}

    for instance_id in tqdm.tqdm(rewriter_output_dict):
        query = rewriter_output_dict[instance_id]["query"]

        if len(query) == 0:
            continue

        query_embedding_dict[instance_id] = model.encode(tokenizer, query)

    with open(rewriter_embedding_path, 'wb') as f:
        pickle.dump(query_embedding_dict, f)