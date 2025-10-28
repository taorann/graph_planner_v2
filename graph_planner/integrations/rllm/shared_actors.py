from __future__ import annotations

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

    def reload_from(
        self,
        model_path: str,
        tensor_parallel_size: int | None = None,
        dtype: str | None = None,
    ) -> str:
        """Hot reload planner weights from ``model_path``."""
        import gc
        from vllm import LLM

        try:
            del self.llm
        except Exception:
            pass
        gc.collect()
        try:
            import torch as _torch

            _torch.cuda.empty_cache()
        except Exception:
            pass

        tp = tensor_parallel_size or int(os.environ.get("PLANNER_TP_SIZE", "2"))
        dt = (dtype or os.environ.get("PLANNER_DTYPE", "bfloat16")).lower()
        dt_map = {"bfloat16": "bfloat16", "bf16": "bfloat16", "float16": "float16", "fp16": "float16"}
        dt_final = dt_map.get(dt, "bfloat16")

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp,
            dtype=dt_final,
            trust_remote_code=True,
        )
        return "ok"


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
        for p in self.model.parameters():
            p.requires_grad_(False)

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
