"""
Generate embedding for code
"""

from transformers import AutoTokenizer, AutoModel
import torch
import os
import numpy as np
import tqdm
import json
import pickle

# custom
import argparse, logging

# input and output path
node_content_path = "xx/node_content/"
node_embedding_path = "xx/tmp_node_embedding/"

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

    node_embedding_dict = None
    node_embedding_list = os.listdir(node_embedding_path)
    node_embedding_list = [item.split('.')[0] for item in node_embedding_list]
    candidate_graphs = os.listdir(node_content_path)

    for filename in tqdm.tqdm(candidate_graphs):

        instance_id = filename.split('.')[0]

        # skip samples which have been processed
        if instance_id in node_embedding_list:
            continue
        
        with open(node_content_path + filename, 'r', encoding='utf-8') as file:
            node_content_dict = json.load(file)
        
        node_list = list(node_content_dict['code'].keys())

        node_embedding_dict = {}
        node_code_embedding_dict = {}
        node_doc_embedding_dict = {}
        for node in node_list:
            code_content = node_content_dict['code'][node]
            if node in node_content_dict['doc']:
                doc_content = node_content_dict['doc'][node]
                code_content = code_content if code_content else "   "
                doc_content = doc_content if doc_content else "   "
                # batch process
                text = [code_content, doc_content]
                node_code_embedding_dict[node], node_doc_embedding_dict[node] = model.encode(tokenizer, text)

            else:
                # for node without doc
                code_content = code_content if code_content else "   "
                node_code_embedding_dict[node] = model.encode(tokenizer, code_content)

        node_embedding_dict = {
            "code": node_code_embedding_dict,
            "doc": node_doc_embedding_dict
        }
        
        with open(node_embedding_path + '{}.pkl'.format(instance_id), 'wb') as f:
            pickle.dump(node_embedding_dict, f)

        torch.cuda.empty_cache()