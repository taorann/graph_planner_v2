from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from vllm.entrypoints.chat_utils import apply_hf_chat_template
import os

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

class QwenAPI:
  def __init__(self, model_path, tensor_parallel_size=4):
    self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)

    sample_params = dict(
      temperature=0.1,
      repetition_penalty=1.1,
      top_k=50,
      top_p=0.98,
      max_tokens=1024
    )

    self.sampling_params = SamplingParams(**sample_params)
    print(f"sampling_params: {sample_params}")

  def get_response(self, system_prompt: str, user_prompt: str):
    conversation = [
      {
        "role": "system",
        "content": system_prompt
      },
      {
        "role": "user",
        "content": user_prompt
      }
    ]
    outputs = self.llm.chat(conversation, sampling_params=self.sampling_params, use_tqdm=False)
    response = outputs[0].outputs[0].text
    return response


if __name__ == "__main__":
  llm = QwenAPI("Qwen/Qwen2.5-1.5B-Instruct")
  user_prompt = "Where is the capital of China?"
  system_prompt = "You are a helpful assistant."
  llm.get_response(system_prompt, user_prompt)