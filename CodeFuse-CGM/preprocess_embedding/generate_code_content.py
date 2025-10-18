"""
Generate node content for all nodes in code graph
"""
from os.path import isfile
import os, sys
import pandas as pd
import json
import tqdm
import re

from codegraph_parser.python.codegraph_python_local import parse, NodeType, EdgeType

def extract_code_and_doc(content):
  """
  split code and doc
  """
  # match docstring
  docstring_pattern = r'"""(.*?)"""|\'\'\'(.*?)\'\'\''
  docstrings = re.findall(docstring_pattern, content, re.DOTALL)

  # extract pure code
  code_without_docstring = re.sub(docstring_pattern, '', content, flags=re.DOTALL)
  # merge docstring
  extracted_docstrings = "\n\n".join([d[0] or d[1] for d in docstrings])
  return code_without_docstring, extracted_docstrings

def get_graph_file_name(item):
    """
    return graph_file_name
    """

    raise NotImplementedError

if __name__ == "__main__":

  graph_basic_df = pd.read_json("test_lite_basic_info.json")
  
  graph_data_path = "codegraph/"
  node_content_path = "xx/node_content/"
  graph_list = os.listdir(graph_data_path)

  # get the graph_file path
  graph_basic_df["graph_file"] = graph_basic_df.apply(lambda item: get_graph_file_name(item), axis=1)

  # generate code content for each repo
  for idx, item in tqdm.tqdm(graph_basic_df.iterrows()):

    instance_id = item.instance_id
    graph_file = item.graph_file
    # get the graph path
    tmp_graph_data_path = graph_data_path + graph_file

    # skip files which have been processed
    if os.path.isfile(node_content_path + '{}.json'.format(instance_id)):
      continue

    graph = parse(tmp_graph_data_path)

    try:
        nodes = graph.get_nodes()
    except:
        print(f"========= parse error: {tmp_graph_data_path} =========")
        continue
    node_code_dict = {}
    node_doc_dict = {}
    for node in nodes:
        node_id = node.node_id
        content = node.get_content()

        code, doc = extract_code_and_doc(content)

        node_code_dict[node_id] = code

        if doc.strip():
            node_doc_dict[node_id] = doc

    # save the result
    with open(node_content_path + '{}.json'.format(instance_id), 'w', encoding='utf-8') as json_file:

        node_content_dict = {
            "code": node_code_dict,
            "doc": node_doc_dict
        }

        json.dump(node_content_dict, json_file, ensure_ascii=False, indent=4)