from typing import List, Dict

def call_model(messages: List[Dict[str,str]]) -> str:
    # TODO: 接入你的 LLM；返回 {"targets":[{path,start,end,confidence,why},...]} 的JSON字符串
    return '{"targets":[]}'  # 占位：让选择器自动回退到规则
