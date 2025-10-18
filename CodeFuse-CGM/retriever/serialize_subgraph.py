"""
serialize subgraph to json file
Here we provide two version
✅ Direct serialize the original subgraph
✅ Serialize the file-level subgraph
"""
import sys
import json
import os
import tqdm
import pandas as pd
import networkx as nx

from codegraph_parser.python.codegraph_python_local import parse, NodeType, EdgeType
from utils import codegraph_to_nxgraph

############################# utils #############################

def get_contained_node(graph_nx, node):
    
    c_node_list = []
    for suc_node in graph_nx.successors(node):
        if graph_nx[node][suc_node][0]['type'] == EdgeType.CONTAINS:
            c_node_list.append(suc_node)
    
    return c_node_list

def get_inner_nodes(graph_nx, node):
    
    inner_nodes = get_contained_node(graph_nx, node)
    inner_nodes_all = []
    
    while len(inner_nodes) != 0:
        
        tmp_inner_nodes = inner_nodes.copy()
        inner_nodes = []
        for node in tmp_inner_nodes:
            inner_nodes_all.append(node)
            inner_nodes.extend(get_contained_node(graph_nx, node))
    
    return list(set(inner_nodes_all))

def serialize_subgraph(graph_nx, file_name):
    
    node_list = [node.to_dict() for node in graph_nx.nodes()]
    
    # 获取子图中的连边关系
    edge_list = []
    for edge in graph_nx.edges():
        edge_type = graph_nx[edge[0]][edge[1]][0]['type']
        tmp_edge_dict = {
            "edgeType": edge_type.name.lower(),
            "source": edge[0].node_id,
            "target": edge[1].node_id
        }
        edge_list.append(tmp_edge_dict)
    # 对所有的边，获取对应边类型
    graph_json = {
        "nodes": node_list, 
        "edges": edge_list
    }

    with open(file_name + '.json', 'w') as json_file:
        json.dump(graph_json, json_file, indent=4)
    
    return True

############################# utils #############################


if __name__ == "__main__":

    test_basic_df = pd.read_json("test_basic_info.json")
    graph_data_path = "codegraph/"
    subgraph_dict_path = "subgraph_nodes.json"
    save_path = "subgraph/"

    with open(subgraph_dict_path, "r", encoding="utf-8") as file:
        one_hop_dict = json.load(file)
        file.close()

    for idx, item in tqdm.tqdm(test_basic_df.iterrows()):

      instance_id = item.instance_id
      graph_file = item.graph_file
      tmp_graph_data_path = graph_data_path + graph_file

      if not os.path.exists(tmp_graph_data_path):
        continue

      filename = save_path + instance_id

      if os.path.exists(filename + '.json'):
        continue

      graph = parse(tmp_graph_data_path)
      graph_nx = codegraph_to_nxgraph(graph)

      # Version 1: Directly Serialization
      # all_nodes = one_hop_dict[instance_id]

      # Version 1: Serialization in File-level
      all_nodes = one_hop_dict[instance_id]
      for node_id in all_nodes:
          node = graph.get_node_by_id(node_id)
          if node.get_type() == NodeType.FILE:
              inner_node = get_inner_nodes(graph_nx, node)
              for i_node in inner_node:
                  all_nodes.append(i_node.node_id)

      all_nodes = [graph.get_node_by_id(node_id) for node_id in all_nodes]
      subgraph = graph_nx.subgraph(list(all_nodes))

      serialize_subgraph(subgraph, filename)