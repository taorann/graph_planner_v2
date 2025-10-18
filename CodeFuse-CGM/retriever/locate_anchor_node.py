"""
基于 rapidfuzz + faiss 进行 anchor node 定位
"""

from rapidfuzz import process, fuzz
import pandas as pd
import json
import tqdm
import sys
import pickle
import numpy as np
import faiss


from codegraph_parser.python.codegraph_python_local import parse, NodeType, EdgeType
from utils import codegraph_to_nxgraph

def extract_info(item):
    """
    抽取需要匹配的字符部分
    """
    return item[1]

################################# Extractor #################################
def get_extractor_anchor(graph, entity_query, keywords_query):
    """
    获取 关键词匹配结果
    """

    all_nodes = graph.get_nodes()

    cand_name_list = []
    cand_path_name_list = []

    for node in all_nodes:
        node_type = node.get_type()
        if node_type in [NodeType.REPO, NodeType.PACKAGE]:
            continue
        
        try:
            node.name
        except:
            continue
        
        cand_name_list.append((node.node_id, node.name))
        
        if node_type == NodeType.FILE:
            if node.path:
                name_with_path = node.path + "/" + node.name
            else:
                name_with_path = node.name
            cand_path_name_list.append((node.node_id, name_with_path))

    cand_name_all = []
    cand_path_name_all = []

    for query in entity_query + keywords_query:
        
        if "/" in query:
            cand_path_name = process.extract((-1, query), cand_path_name_list, scorer=fuzz.WRatio, limit=3, processor=extract_info)
            cand_path_name_all.append(cand_path_name)

        query_wo_path = query.split('/')[-1]
        cand_name = process.extract((-1, query_wo_path), cand_name_list, scorer=fuzz.WRatio, limit=3, processor=extract_info)
        cand_name_all.append(cand_name)
            

    res = set()
    for query in cand_name_all:
        for item in query:
            res.add(item[0][0])
    for query in cand_path_name_all:
        for item in query:
            res.add(item[0][0])

    return res

################################# Extractor #################################

################################# Inferer #################################
def get_inferer_anchor(query_emb, node_embedding, k=15):
    """
    根据 embedding 进行语义检索
    """
    
    node2id_dict = {}
    id2node_dict = {}
    cand_vec = []

    raw_node_embedding = node_embedding["code"]
    for i, node_id in enumerate(raw_node_embedding):
        node2id_dict[node_id] = i
        id2node_dict[i] = node_id
        cand_vec.append(raw_node_embedding[node_id])

    cand_vec_np = np.array(cand_vec)

    ######### search #########
    d = 1024
    nb = len(cand_vec_np)
    nq = 5

    index = faiss.IndexFlatL2(d)
    index.add(cand_vec_np)
    D, I = index.search(cand_vec_np[:5], k)
    D, I = index.search(query_emb, k)

    anchor_node = []
    for query in I:
        tmp_node_list = []
        for trans_id in query:
            tmp_node_list.append(int(id2node_dict[trans_id]))
        anchor_node.append(tmp_node_list)

    return anchor_node

    
################################# Inferer #################################

################################# 辅助函数 #################################
def get_graph_file_name(item):
    """
    return graph_file_name
    """

    raise NotImplementedError
################################# 辅助函数 #################################


if __name__ == "__main__":

    # 数据变量定义
    test_basic_df = pd.read_json("test_basic_info.json")
    test_basic_df["graph_file"] = test_basic_df.apply(lambda item: get_graph_file_name(item), axis=1)
    
    graph_data_path = "codegraph/" # the path of codegraphs

    # 读入 rewriter 提取结果 和 node embedding
    rewriter_output_path = "/rewriter_output.json"
    query_embedding_path = "/rewriter_embedding.pkl"
    node_embedding_path = "/node_embedding/"
    with open(rewriter_output_path, "r", encoding="utf-8") as file:
        rewriter_output = json.load(file)
        file.close()
    
    with open(query_embedding_path, "rb") as file:
        query_embedding = pickle.load(file)
        file.close()
    
    # save path
    anchor_node_dict = {}

    for idx, item in tqdm.tqdm(test_basic_df.iterrows()):

        instance_id = item.instance_id
        graph_file = item.graph_file
        tmp_graph_data_path = graph_data_path + graph_file
        query_emb = query_embedding[instance_id]

        # 解析图数据
        graph = parse(tmp_graph_data_path)
        graph_nx = codegraph_to_nxgraph(graph)

        # 获取 rewriter 输出
        entity_query = rewriter_output[instance_id]["code_entity"]
        keyword_query = rewriter_output[instance_id]["keyword"]

        # 读入 node_embedding
        tmp_node_embedding = node_embedding_path + "{}.pkl".format(instance_id)
        with open(tmp_node_embedding, "rb") as file:
            tmp_node_embedding = pickle.load(file)
            file.close()

        # 定位 anchor nodes
        res_extractor = get_extractor_anchor(graph, entity_query, keyword_query)
        res_inferer = get_inferer_anchor(query_emb, tmp_node_embedding)

        anchor_node = {
            "extractor_anchor_nodes": list(res_extractor),
            "inferer_anchor_nodes": list(res_inferer),
        }

        anchor_node_dict[instance_id] = anchor_node
        
        # save
        with open("anchor_node.json", 'w', encoding='utf-8') as file:
            json.dump(anchor_node_dict, file)
