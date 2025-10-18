import json
import json
# from codegraph import *

from torch.utils.data.dataset import Dataset
from transformers import AutoModel, AutoTokenizer

# from FlagEmbedding import BGEM3FlagModel
# from sentence_transformers import SentenceTransformer

from datasets import Dataset as HFDataset
from datasets import load_dataset

import torch
import numpy as np
import logging
import time
import gc

import random
import string
import os
import sys

import json
from collections import defaultdict
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getJavaSentence(node, nodeType, reponame, max_len):
    def process_Repo(node):
        return reponame

    def process_Module(node):
        return node['name']

    def process_Package(node):
        return node['name']

    def process_File(node):
        path = node.get('path','')
        if len(path) > 0:
            path = path + '/'
        return f"{path}{node['name']}"

    def process_TextFile(node):
        return f"{node['name']}\n{node.get('text','')}"

    def process_Class(node):
        return f"{node.get('modifiers','')} {node['name']}\n{node.get('comment','')}".strip(' ')

    def process_Field(node):
        return f"{node.get('modifiers','')} {node['fieldType']} {node['name']}\n{node.get('comment','')}".strip(' ')

    def process_Method(node):
        className = node.get('className','')
        methodName = node.get('methodName', '')
        if len(methodName) == 0 or len(className) == 0:
            split = node['signature'].split('#')
            className = split[0]
            methodName = split[1].split('(')[0]
        name = className + '.' + methodName
        comment = f"{node.get('comment','')}\n" if not node.get('comment','') == '' else ''
        text = f"{node.get('modifiers','')} {node.get('text','')}" if not node.get('modifiers','') == '' else node.get('text','')
        return f"{name}\n{comment}{text}"

    def process_default(node):
        raise ValueError(f"unrecognized nodeType for node.keys {node['nodeType']} {str(node.keys())}")
        return ""

    processors = {
        'Repo': process_Repo,
        'Module': process_Module,
        'Package': process_Package,
        'File': process_File,
        'TextFile': process_TextFile,
        'Textfile': process_TextFile,
        'Class': process_Class,
        'Field': process_Field,
        'Method': process_Method
    }

    sentence = processors.get(nodeType, process_default)(node)

    # TODO: limit token not str size
    if len(sentence) > max_len:
        sentence = sentence[:max_len]

    return sentence

def getPythonSentence(node, nodeType, reponame, max_len):
    def process_Repo(node):
        return reponame

    def process_Package(node):
        return node['name']

    def process_File(node):
        path = node.get('filePath','')
        if len(path) > 0:
            path = path + '/'
        return f"{path}{node['fileName']}\n{node.get('text','')}"

    def process_TextFile(node):
        return f"{node['name']}\n{node.get('text','')}"

    def process_Class(node):
        return f"{node.get('classType','')} {node['className']}\n{node.get('comment','')}\n{node.get('text','')}".strip(' ')

    def process_Attribute(node):
        return f"{node.get('attributeType','')} {node['name']}\n{node.get('comment','')}\n{node.get('text','')}".strip(' ')

    def process_Function(node):
        comment = f"{node.get('comment','')}\n" if not node.get('comment','') == '' else ''
        return f"{node.get('header','')} {node['name']}\n{comment}{node.get('text','')}".strip(' ')

    def process_Lambda(node):
        return f"{node.get('text','')}".strip(' ')

    def process_default(node):
        raise ValueError(f"unrecognized nodeType for node.keys {node['nodeType']} {str(node.keys())}")
        return ""

    processors = {
        'Repo': process_Repo,
        'Package': process_Package,
        'File': process_File,
        'TextFile': process_TextFile,
        'Textfile': process_TextFile,
        'Class': process_Class,
        'Attribute': process_Attribute,
        'Function': process_Function,
        'Lambda': process_Lambda
    }

    sentence = processors.get(nodeType, process_default)(node)

    # TODO: limit token not str size
    if len(sentence) > max_len:
        sentence = sentence[:max_len]

    return sentence

def graph2embedding(data, model, tokenizor, reponame, language, save_adj):
    node_embeddings = {}
    sentence_dict = {}
    node_id_to_index = {}
    index_counter = 0

    for node in data['nodes']:
        nodeType = node['nodeType']

        if 'nodeId' in node.keys():
            node_id = node['nodeId']
        elif 'id' in node.keys():
            node_id = node['id']
        else:
            raise ValueError("No key named id/nodeId")

        if language == 'java':
            sentence = getJavaSentence(node, nodeType, reponame, 1024000)
        elif language == 'python':
            sentence = getPythonSentence(node, nodeType, reponame, 1024000)
        else:
            raise ValueError(f"Language {language} not supported")

        if sentence == "":
            node_embedding = torch.zeros((1, 256), dtype=torch.float32).to(device)
            node_embeddings[node_id] = [node_embedding]
            sentence_dict[index_counter] = ""
            node_id_to_index[node_id] = [index_counter]
            index_counter += 1
        else:
            # 手动切词
            tokens = tokenizor.tokenize(sentence)
            num_tokens = len(tokens)
            num_segments = (num_tokens + 511) // 512  # Calculate number of segments
            embeddings = []
            segments = []
            node_id_to_index[node_id] = list(range(index_counter, index_counter + num_segments))
            for i in range(num_segments):
                start = i * 512
                end = min((i + 1) * 512, num_tokens)
                segment_tokens = tokens[start:end]
                segment_sentence = tokenizor.convert_tokens_to_string(segment_tokens)
                segment_ids = tokenizor.encode(segment_sentence, return_tensors="pt").to(device)
                with torch.no_grad():
                    segment_embedding = model(segment_ids)
                embeddings.append(segment_embedding)
                segments.append(segment_sentence)
                sentence_dict[index_counter] = segment_sentence
                index_counter += 1

            node_embeddings[node_id] = embeddings

    num_nodes = index_counter
    
    if save_adj:
        adj_matrix = torch.zeros((num_nodes, num_nodes))

        for edge in data['edges']:
            source_id = edge['source']
            target_id = edge['target']
            source_indices = node_id_to_index.get(source_id)
            target_indices = node_id_to_index.get(target_id)
            if source_indices is None or target_indices is None:
                # if source_indices is None: 
                #     print(f"{source_id} not exists")
                # if target_indices is None:
                #     print(f"{target_id} not exists")
                continue
            
            for source_index in source_indices:
                for target_index in target_indices:
                    adj_matrix[source_index, target_index] = 1

        # Connect embeddings of the same node
        for node_id, indices in node_id_to_index.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    adj_matrix[indices[i], indices[j]] = 1
                    adj_matrix[indices[j], indices[i]] = 1
    else:
        adj_matrix = None

    all_embeddings = []
    for value in node_embeddings.values():
        if isinstance(value, torch.Tensor):
            all_embeddings.append(value)
        elif isinstance(value, list):
            for tensor in value:
                all_embeddings.append(tensor)

    embeddings = torch.stack(all_embeddings, dim=0)

    # embeddings = torch.stack(list(node_embeddings.values()))
    # embeddings = torch.stack(sum(node_embeddings.values(), []))
    # embeddings = torch.cat(list(node_embeddings.values()), dim=0)

    return embeddings, adj_matrix, sentence_dict

def preprocess_graph(graphdir, savedir, recdir, jsondir, language = 'java', model = None, tokenizor = None, filenum = 1, suffix = 'pt', node_limit = 20000, save_adj = True, save_rec = True):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Parsing json file: {jsondir}{filenum}.json")

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if not os.path.exists(recdir):
        os.makedirs(recdir)

    if jsondir == graphdir:
        glist = os.listdir(graphdir)
    else:
        with open(f'{jsondir}{filenum}.json', 'r') as f:
            glist = json.load(f)
        f.close()

    for gname in glist:
        if gname.startswith('._'):
            gname = gname[2:]
        raw_file = os.path.join(graphdir, gname)
        print(gname)

        if len(gname.split('#')) == 2:
            appName = gname.split('#')[1].split('-graph.json')[0]
            repoName = appName
            groupName = gname.split('#')[0]
            commitId = '0'
        elif len(gname.split('#')) == 3:
            appName = gname.split('#')[1]
            repoName = appName
            groupName = gname.split('#')[0]
            commitId = gname.split('#')[2].split('.graph.json')[0]
        elif len(gname.split('___')) == 3:
            parts = gname.split('___')
            appName = parts[0]
            repoName = parts[1].split('__')[1]
            groupName = parts[1].split('__')[0]
            commitId = parts[2].split('.')[0]
        else:
            print(f"{gname} can't be renamed")
            continue
        tmp1 = f"{appName}___{repoName}___{groupName}___{commitId}.{suffix}"
        tmp2 = f"{appName}___{repoName}___{groupName}___{commitId}.json"
        print(tmp1)

        save_file = os.path.join(savedir, tmp1)
        rec_file = os.path.join(recdir, tmp2)

        if not os.path.exists(raw_file):
            continue
        if os.path.exists(save_file) and os.path.exists(rec_file):
            continue
        logger.info(f'Start {gname} transforming...')
        try:
            with open(raw_file, 'r') as f1:
                content = f1.read()
                data = json.loads(content)

                if len(data['nodes']) > node_limit:
                    continue
                embeddings, adj_matrix, sentence_dict = graph2embedding(data, model, tokenizor, gname, language, save_adj)
            f1.close()

            if suffix == 'json':
                if save_adj:
                    data_dict = {
                        "embeddings": embeddings.tolist(),
                        "adj_matrix": adj_matrix.tolist()
                    }
                else:
                    data_dict = {
                        "embeddings": embeddings.tolist(),
                    }

                with open(save_file, 'w') as f:
                    json.dump(data_dict, f)
                f.close()

            elif suffix == 'pt':
                if save_adj:
                    data_dict = {
                        "embeddings": embeddings.detach(),
                        "adj_matrix": adj_matrix.detach()
                    }
                else:
                    data_dict = {
                        "embeddings": embeddings.detach(),
                    }
                torch.save(data_dict, save_file)

            if save_rec:
                rec_dict = {
                    "text": list(sentence_dict.values())
                }

                with open(rec_file, 'w') as f:
                    json.dump(rec_dict, f)
                f.close()
        except json.JSONDecodeError as e:
            print('Json Decode Error: '+ gname)

def preprocess(graphdir, savedir, recdir, jsondir, language = 'java', mode = 'pretrain', filenum = 1, suffix = 'pt', node_limit = 20000, save_adj = True, save_rec = True):

    model1_path = "salesforce/codet5p-110m-embedding"
    tokenizer1 = AutoTokenizer.from_pretrained(model1_path, trust_remote_code=True, device = device)
    model1 = AutoModel.from_pretrained(model1_path, trust_remote_code=True, torch_dtype="auto").to(device).eval()

    if mode == 'pretrain':
        preprocess_graph(
            graphdir=graphdir, 
            savedir=savedir, 
            recdir=recdir, 
            jsondir=jsondir, 
            language=language,
            model=model1, 
            tokenizor=tokenizer1, 
            filenum=filenum, 
            suffix=suffix, 
            node_limit=node_limit,
            save_adj=save_adj,
            save_rec=save_rec)
    else:
        raise NotImplementedError

def json_split(loaddirs, savedir, split_num=64):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    file_list = []
    for loaddir in loaddirs:
        file_list += os.listdir(loaddir)
    total_num = len(file_list)
    sep_num = total_num // split_num
    print(f'total num: {total_num}, sep num: {sep_num}')

    for i in range(split_num):
        start = i * sep_num
        end = start + sep_num if i != split_num - 1 else total_num
        with open(f'{savedir}/{i+1}.json', 'w') as f:
            json.dump(file_list[start:end], f)
        f.close()

def json_split_from_json(input_json, savedir, split_num=64):
    with open(input_json, 'r') as file:
        data = json.load(file)

    total_items = len(data)
    num_files = (total_items + split_num - 1) // split_num  # 向上取整

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for i in range(num_files):
        start = i * split_num
        end = min(start + split_num, total_items)
        split_data = data[start:end]

        save_file = os.path.join(savedir, f"{i+1}.json")

        with open(save_file, 'w') as file:
            json.dump(split_data, file, indent=4)

def detect_pt_file_errors(directory, output_json):
    error_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        tmp = torch.load(f)
                        del tmp
                    gc.collect()
                except Exception as e:
                    error_files.append(file_path)
                    print(f"Error loading {file_path}: {e}")

    with open(output_json, 'w') as f:
        json.dump(error_files, f, indent=4)

    print(f"Detected {len(error_files)} error files. Details saved in {output_json}")

def transfer_pt_file_errors(input_json, output_json):
    with open(input_json, 'r') as file:
        data = json.load(file)

    def transform_path(path):
        repo_str = path.split('/')[-1]
        appName = repo_str.split('___')[0]
        groupName = repo_str.split('___')[2]
        new_repo_str = f"{groupName}#{appName}-graph.json"
        return new_repo_str

    transformed_data = [transform_path(path) for path in data]

    with open(output_json, 'w') as file:
        json.dump(transformed_data, file, indent=4)

def get_list(graph_dirs):
    all_files = [os.path.join(graph_dir, file) 
                 for graph_dir in graph_dirs 
                 for file in os.listdir(graph_dir)]
    return all_files

def get_list_constrained(graph_dirs, size_limit = 500 * 1024 * 1024):  
    filtered_files = []
    for graph_dir in graph_dirs:
        glist = os.listdir(graph_dir)
        for file_name in glist:
            file_path = os.path.join(graph_dir, file_name)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                if file_size < size_limit:
                    filtered_files.append(file_path)

    return filtered_files

def get_graph_path(glist, filename, suffix):
    sp = filename.split('___')
    if len(sp) == 4:
        appName = sp[0]
        repoName = sp[1]
        groupName = sp[2]
        commitId = sp[3].split('.')[0]

        matched_graphs = []
        for graph in glist:
            graph_parts = graph.split('/')[-1].split('___')
            if len(graph_parts) == 4:
                graph_appName = graph_parts[0]
                graph_repoName = graph_parts[1]
                graph_groupName = graph_parts[2]
                graph_commitId = graph_parts[3].split('.')[0]

                if graph_appName == appName:
                    matched_graphs.append((graph, graph_repoName, graph_groupName, graph_commitId))

        if not matched_graphs:
            return None

        if not commitId == '0':
            for graph, graph_repoName, graph_groupName, graph_commitId in matched_graphs:
                if commitId == graph_commitId:
                    return graph

        best_match = None
        best_match_score = -2
        for graph, graph_repoName, graph_groupName, _ in matched_graphs:
            score = (repoName == graph_repoName) + (groupName == graph_groupName)
            if score > best_match_score:
                best_match_score = score
                best_match = graph
        
        return best_match
    else:
        raise ValueError(f"{filename} to graph not supported")

def split_jsonl_dataset(input_file, train_file, test_file, train_ratio=0.98):
    def read_jsonl(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                yield json.loads(line)

    data = list(read_jsonl(input_file))
    repo_dict = defaultdict(list)

    for item in data:
        repo_dict[item['repo']].append(item)

    repos = list(repo_dict.keys())
    random.shuffle(repos)

    split_index = int(len(repos) * train_ratio)
    train_repos = repos[:split_index]
    test_repos = repos[split_index:]
    
    train_data = []
    test_data = []

    for repo in train_repos:
        train_data.extend(repo_dict[repo])
    for repo in test_repos:
        test_data.extend(repo_dict[repo])

    with open(train_file, 'w') as file:
        for item in train_data:
            file.write(json.dumps(item) + '\n')
    
    with open(test_file, 'w') as file:
        for item in test_data:
            file.write(json.dumps(item) + '\n')


