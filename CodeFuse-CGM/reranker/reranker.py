import json
import re
import os
import pandas as pd
import argparse

from qwen_api import QwenAPI
from codegraph_parser.python.codegraph_python_local import parse, NodeType, EdgeType

stage_1_system_prompt = """
You are an experienced software developer who specializes in extracting the most relevant files for solving issues from many reference files.

Task:
Based on the information received about the issue from a repository, find the most likely few files from among those that may be able to resolve the issue.

Instructions:
1. Analysis:
- Analyze the provided issue description and files, and pay attention to the relevance of the provided files with the given issue, especially those might be modified during fixing the issue.
- Determine the specific problem or error mentioned in the issue and note any clues that could help your judgment.
2. Extraction:
- Based on your analysis, choose the Top **{}** relevant files which might be used in fixing the issue.
- You should choose files from the provided files, and should not modify their name in any way.

Respond in the following format:
[start_of_analysis]
<detailed_analysis> 
[end_of_analysis] 

[start_of_relevant_files] 
1. <file_with_its_path>
2. <file_with_its_path>
3. ...
[end_of_relevant_files] 

Notes:
- You can refer to to the information in the error logs (if exists).
- The relevant file usually exists in the project described in the issue (e.g., django, sklearn). File need modification is usually not in the tests files or external packages.
- The file you choose should be contained in the provided files.
- Provide the file path with files. Do not include redundant suffix like '/home/username/', '/etc/service/' or '/tree/master'.
- Do not include any additional information such as line numbers or explanations in your extraction result.
- Files for initialization and configuration might be modified during changing the code.

Preferred extraction Examples of Related Files:
1. src/utils/file_handler.py
2. core/services/service_manager.py
3. ...
""".strip()

stage_1_user_prompt_template = """
<repository>
{}
</repository>

<issue>
{}
</issue>
 
<reference_python_file_list>
{}
</reference_python_file_list>

<other_reference_file_list>
{}
</other_reference_file_list>
"""

stage_2_system_prompt_v3 = """
You are an experienced software developer who specializes in assessing the relevance of the file for solving the issue in software repositories.

Task:
For a file provided, evaluate the likelihood that modifying this file would resolve the given issue, and assign a score based on specific criteria.

Instructions:
1. Analysis:
- Analyze the provided issue description and the content of the single relevant file, pay attention to any keywords, error messages, or specific functionalities mentioned that relate to the file.
- Determine how closely the contents and functionality of the file are tied to the problem or error described in the issue.
- Consider the role of the file in the overall project structure (e.g., configuration files, core logic files versus test files, or utility scripts).
2. Scoring:
- Based on your analysis, assign a score from 1 to 5 that represents the relevance of modifying the given file in order to solve the issue.

Score Specifications:
1. **Score 1**: The file is almost certainly unrelated to the issue, with no apparent connection to the functionality or error described in the issue.
2. **Score 2**: The file may be tangentially related, but modifying it is unlikely to resolve the issue directly; possible in rare edge cases.
3. **Score 3**: The file has some relevance to the issue; it might interact with the affected functionality indirectly and tweaking it could be part of a broader fix.
4. **Score 4**: The file is likely related to the issue; it includes code that interacts directly with the functionality in question and could plausibly contain bugs that lead to the issue.
5. **Score 5**: The file is very likely the root cause or heavily involved in the issue and modifying it should directly address the error or problem mentioned.

Respond in the following format:
[start_of_analysis]
<detailed_analysis>
[end_of_analysis]

[start_of_score]
Score <number>
[end_of_score]

Notes:
- The content of the file shows only the structure of this file, including the names of the classes and functions defined in this file.
- You can refer to to the information in the error logs (if exists).
""".strip()

stage_2_user_prompt_template = """
<repository>
{}
</repository>

<issue>
{}
</issue>

<file_name>
{}
</file_name>

<file_content>
{}
</file_content>
"""


def get_python_inner_class_and_function(graph, node_id, layer_cnt = 0):
    """
    寻找某个node的内部函数和类的node，返回list
    dfs，返回的列表每个item是(深度, node)
    """
    ret_list = []

    # 限制深度
    if layer_cnt > 5:
        return ret_list

    node = graph.get_node_by_id(node_id)
    inner_node_ids = graph.get_out_nodes(node_id)
    for inner_node_id in inner_node_ids:
        inner_node = graph.get_node_by_id(inner_node_id)

        if inner_node.get_type() == NodeType.FUNCTION and "def " + inner_node.name in node.text:
            ret_list.append((layer_cnt, inner_node))
            ret_list.extend(get_python_inner_class_and_function(graph, inner_node.node_id, layer_cnt + 1))
        elif inner_node.get_type() == NodeType.CLASS and "class " + inner_node.name in node.text:
            ret_list.append((layer_cnt, inner_node))
            ret_list.extend(get_python_inner_class_and_function(graph, inner_node.node_id, layer_cnt + 1))
    
    return ret_list

def parse_reranker_stage_1(response):
    # relevant_file
    pattern = r"\[start_of_relevant_files\]\s*(.*?)\s*\[end_of_relevant_files\]"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        relevant_files = match.group(1).strip().split("\n")
    else:
        pattern = r"<start_of_relevant_files>\s*(.*?)\s*<end_of_relevant_files>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            relevant_files = match.group(1).strip().split("\n")
        else:
            pattern = r"\[Start_of_Relevant_Files\]\s*(.*?)\s*\[End_of_Relevant_Files\]"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                relevant_files = match.group(1).strip().split("\n")
            else:
                relevant_files = []

    print(relevant_files)
    for idx, relevant_file in enumerate(relevant_files):
        new_relevant_file = relevant_file
        if new_relevant_file.startswith("- "):
            new_relevant_file = new_relevant_file[2:]

        pattern = r"\d+ *\.(.+)"
        match = re.search(pattern, new_relevant_file)
        if match:
            new_relevant_file = match.group(1).strip()
        relevant_files[idx] = new_relevant_file
    
    return relevant_files

def parse_reranker_stage_2(response):
    # score
    pattern = r"\[start_of_score\]\s*(.*?)\s*\[end_of_score\]"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        score = match.group(1).strip().split("\n")
    else:
        pattern = r"<start_of_score>\s*(.*?)\s*<end_of_score>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            score = match.group(1).strip().split("\n")
        else:
            pattern = r"\[Start_of_Score\]\s*(.*?)\s*\[End_of_Score\]"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                score = match.group(1).strip().split("\n")
            else:
                score = ["0"]

    score = score[0]
    if score.startswith("- "):
        score = score[2:]

    pattern = r"Score (\d+)"
    match = re.search(pattern, score)
    if match:
        score = match.group(1)
        score = int(score)
    else:
        score = 0

    return score

def extract_files_from_subgraph(subgraph_path, output_path):
    subgraph_list = os.listdir(subgraph_path)
    # print(len(subgraph_list))

    for subgraph in subgraph_list:
        if not subgraph.endswith(".json"):
            continue
        # print(subgraph)
        try:
            with open(os.path.join(subgraph_path, subgraph), "r", encoding="utf-8") as f:
                subgraph_json = json.load(f)
        except:
            print(f"broken json file: {subgraph}")
            continue
        subgraph_nodes = subgraph_json["nodes"]
        file_nodes = [node for node in subgraph_nodes if node["nodeType"] == "File"]
        pred_files = []
        for node in file_nodes:
            file_path = node["filePath"]
            file_name = node["fileName"]
            if file_path is None:
                file = file_name
            else:
                file = os.path.join(file_path, file_name)
            pred_files.append(file)

        subgraph_name = subgraph.split(".")[0]
        with open((os.path.join(output_path, subgraph_name + ".json")), "w", encoding="utf-8") as f:
            json.dump(pred_files, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Reranker.")
    
    parser.add_argument('--stage_1_k', type=int, default=10, help='Specify the k for stage 1')
    parser.add_argument('--stage_2_k', type=int, default=5, help='Specify the k for stage 2')
    
    return parser.parse_args()


if __name__ == "__main__":
    llm = QwenAPI("Qwen/Qwen2.5-72B-Instruct")

    output_dir = "reranker_outputs/"

    subgraph_file_dir = "subgraph/"

    # retriever得到的file list
    retriever_filtered_files_dir = "subgraph_extracted_files/"
    os.makedirs(retriever_filtered_files_dir, exist_ok=True)
    extract_files_from_subgraph(subgraph_file_dir, retriever_filtered_files_dir)

    df = pd.read_json("test_basic_info.json")

    args = parse_args()

    # stage_1
    stage_1 = True
    stage_1_output_dir = os.path.join(output_dir, f"stage_1_top_{args.stage_1_k}")
    os.makedirs(os.path.join(stage_1_output_dir, "relevant_files"), exist_ok=True)
    os.makedirs(os.path.join(stage_1_output_dir, "response"), exist_ok=True)
    stage_1_system_prompt = stage_1_system_prompt.format(args.stage_1_k)
    if stage_1:
        reranker_stage_1_outputs = os.listdir(os.path.join(stage_1_output_dir, "relevant_files"))
        reranker_stage_1_outputs = [item.split(".")[0] for item in reranker_stage_1_outputs]
        for i, data in enumerate(df):
            repo, instance_id, base_commit, patch, test_patch, problem_statement, hints_text, created_at, version, fail_to_pass, pass_to_pass = data["repo"], data["instance_id"], data["base_commit"], data["patch"], data["test_patch"], data["problem_statement"], data["hints_text"], data["created_at"], data["version"], data["FAIL_TO_PASS"], data["PASS_TO_PASS"]
            
            if instance_id in reranker_stage_1_outputs:
                print(f"Stage 1 index {i} skip")
                continue
            
            if os.path.exists(os.path.join(retriever_filtered_files_dir, instance_id + ".json")):
                with open(os.path.join(retriever_filtered_files_dir, instance_id + ".json"), "r") as f:
                    filtered_files = json.load(f)
            else:
                raise ValueError
            
            python_files = [item for item in filtered_files if item.endswith(".py")]
            other_files = [item for item in filtered_files if not item.endswith(".py")]
            user_prompt = stage_1_user_prompt_template.format(repo, problem_statement, "\n".join(python_files), "\n".join(other_files))
            print(user_prompt)

            response = llm.get_response(stage_1_system_prompt, user_prompt)
            print(response)

            relevant_files = parse_reranker_stage_1(response)
            with open(os.path.join(stage_1_output_dir, "relevant_files", instance_id + ".json"), "w") as f:
                json.dump(relevant_files, f, indent=4)
            with open(os.path.join(stage_1_output_dir, "response", instance_id + ".txt"), "w", encoding="utf-8") as f:
                f.write(response)

            print(f"Stage 1 index {i} done")

    # stage_2
    stage_2 = True
    stage_2_output_dir = os.path.join(output_dir, f"stage_2_{args.stage_2_k}")
    os.makedirs(os.path.join(stage_2_output_dir, "relevant_files"), exist_ok=True)
    os.makedirs(os.path.join(stage_2_output_dir, "response"), exist_ok=True)
    if stage_2:
        reranker_stage_2_outputs = os.listdir(os.path.join(stage_2_output_dir, "relevant_files"))
        reranker_stage_2_outputs = [item.split(".")[0] for item in reranker_stage_2_outputs]
        for i, data in enumerate(df):
            repo, instance_id, base_commit, patch, test_patch, problem_statement, hints_text, created_at, version, fail_to_pass, pass_to_pass = data["repo"], data["instance_id"], data["base_commit"], data["patch"], data["test_patch"], data["problem_statement"], data["hints_text"], data["created_at"], data["version"], data["FAIL_TO_PASS"], data["PASS_TO_PASS"]
            
            if instance_id in reranker_stage_2_outputs:
                print(f"Stage 2 index {i} skip")
                continue
            
            with open(os.path.join(stage_1_output_dir, "relevant_files", instance_id + ".json"), "r") as f:
                stage_1_relevant_files = json.load(f)
            
            # 读取子图
            if os.path.exists(os.path.join(subgraph_file_dir, instance_id + ".json")):
                subgraph_file_path = os.path.join(subgraph_file_dir, instance_id + ".json")
                graph = parse(subgraph_file_path)
            else:
                raise ValueError

            relevant_file_score = {}
            relevant_file_response = {}
            for relevant_file in stage_1_relevant_files:
                relevant_file_content = ""
                find = False

                # 保留class和function
                for file_node in graph.get_nodes_by_type(NodeType.FILE):
                    file_path = file_node.path
                    file_name = file_node.name
                    if file_path is None:
                        file = file_name
                    else:
                        file = os.path.join(file_path, file_name)
                    if file == relevant_file:
                        class_and_function_list = get_python_inner_class_and_function(graph, file_node.node_id)

                        relevant_file_content = ""
                        for layer, node in class_and_function_list:
                            # 添加缩进
                            if node.get_type() == NodeType.CLASS:
                                relevant_file_content += "    " * layer + "class " + node.name  # 接口文档中的字段是className，但实际parse时会处理到name中                                
                                relevant_file_content += "\n"
                            elif node.get_type() == NodeType.FUNCTION:
                                if node.name != "<anonymous>":
                                    relevant_file_content += "    " * layer + "def " + node.name
                                    relevant_file_content += "\n"

                        find = True
                        break

                # 未找到直接默认0，不打分
                if not find:
                    relevant_file_score[relevant_file] = 0
                    relevant_file_response[relevant_file] = ""
                else:
                    user_prompt = stage_2_user_prompt_template.format(repo, problem_statement, relevant_file, relevant_file_content)
                    print(user_prompt)

                    response = llm.get_response(stage_2_system_prompt_v3, user_prompt)

                    score = parse_reranker_stage_2(response)
                    relevant_file_score[relevant_file] = score
                    relevant_file_response[relevant_file] = response

            k = args.stage_2_k
            # 找出分数最高的topk个文件
            sorted_relevant_files = sorted(relevant_file_score.items(), key=lambda item: item[1], reverse=True)
            if k <= len(sorted_relevant_files):
                selected_relevant_files = sorted_relevant_files[:k]
                selected_relevant_files = [item[0] for item in selected_relevant_files]
            else:
                selected_relevant_files = [item[0] for item in sorted_relevant_files]
            
            with open(os.path.join(stage_2_output_dir, "relevant_files", instance_id + ".json"), "w") as f:
                json.dump(dict(relevant_file_score=relevant_file_score, selected_relevant_files=selected_relevant_files), f, indent=4)
            with open(os.path.join(stage_2_output_dir, "response", instance_id + ".json"), "w") as f:
                json.dump(relevant_file_response, f, indent=4)

            print(f"Stage 2 index {i} done")

