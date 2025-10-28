import os
import ray
import torch
from typing import List, Dict


@ray.remote(
    num_gpus=2,
    max_concurrency=64,
    runtime_env={"env_vars": {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "VLLM_USE_V1": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    }}
)
class PlannerEngine:
    def __init__(self):
        from vllm import LLM

        mp = os.environ["PLANNER_MODEL_PATH"]
        self.llm = LLM(
            model=mp,
            tensor_parallel_size=2,
            dtype="bfloat16",
            trust_remote_code=True,
        )

    def generate(self, prompts: List[str], **kw) -> List[str]:
        from vllm import SamplingParams

        sp = SamplingParams(
            temperature=float(os.environ.get("PLANNER_MODEL_TEMPERATURE", "0.2")),
            top_p=float(os.environ.get("PLANNER_MODEL_TOP_P", "0.95")),
            max_tokens=int(os.environ.get("PLANNER_MODEL_MAX_TOKENS", "512")),
        )
        outs = self.llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outs]


@ray.remote(
    num_gpus=2,
    max_concurrency=64,
    runtime_env={"env_vars": {
        "CUDA_VISIBLE_DEVICES": "2,3",
    }}
)
class CGMTool:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        mp = os.environ["CGM_MODEL_PATH"]
        tp = os.environ.get("CGM_TOKENIZER_PATH", mp)
        self.tok = AutoTokenizer.from_pretrained(tp, use_fast=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            mp, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.model.eval()

    @torch.inference_mode()
    def chat(self, messages: List[Dict], **gen_kw) -> str:
        prompt = self._build_prompt(messages)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=int(os.environ.get("CGM_MAX_TOKENS", "768")),
            do_sample=True,
            temperature=float(os.environ.get("CGM_TEMPERATURE", "0.2")),
            top_p=float(os.environ.get("CGM_TOP_P", "0.9")),
        )
        return self.tok.decode(out[0], skip_special_tokens=True)

    def _build_prompt(self, messages: List[Dict]) -> str:
        parts = []
        for m in messages:
            parts.append(f"{m.get('role','user').upper()}: {m.get('content','')}")
        return "\n".join(parts) + "\nASSISTANT:"
