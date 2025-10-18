from dataclasses import dataclass, asdict
import argparse, json
from typing import List, Union
import torch


@dataclass
class TrainArgs:
    graph_dir: Union[str, List[str]]
    train_files: Union[str, List[str]]
    valid_files: Union[str, List[str]]
    output_dir: str
    tb_dir: str

    embedding_dim: int = 2304

    load_pretrained_encoder: bool = False
    pretrained_encoder_path: Union[None, str] = None
    load_pretrained_adapter: bool = False
    pretrained_adapter_path: Union[None, str] = None
    adapter_hidden_dim: int = 4096
    adapter_num_layers: int = 1
    adapter_num_heads: int = 8

    self_defined: bool = False
    pretrained_model_path: Union[None, str] = None
    lm_hidden_dim: int = 4096
    quantization: Union[None, str] = None
    framework_type: Union[None, str] = "default"
    model_type: Union[None, str] = None

    load_pretrained_tokenizer: bool = True
    pretrained_tokenizer_path: Union[None, str] = None

    # for evaluation
    pretrained_lora_path: Union[None, str] = None

    # training mode:
    #  "e" 1, "a" 2, "l" 3
    mode: str = "a"
    task: str = "align"
    use_chat: bool = True
    use_adj: bool = False

    # lora rank, the bigger, the more trainalbe parameters
    peft: Union[None, str] = None
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_modules: Union[str, List[str]] = "all-linear"

    enc_peft: Union[None, str] = None
    enc_lora_rank: int = 32
    enc_lora_alpha: int = 32
    enc_lora_dropout: float = 0.05
    enc_lora_modules: Union[str, List[str]] = "all-linear"

    graph_pad_token: str = "<｜graph_pad｜>"
    graph_pad_id: int = 32022
    graph_token_num: int = 512

    learning_rate: float = 5e-5
    min_lr: float = 5e-6
    weight_decay: float = 0.1
    lr_scheduler_type: str = "cosine"

    gradient_accumulation_steps: int = 1
    num_warmup_steps: int = 300
    adapter_warmup: bool = False
    adapter_warmup_steps: int = 500
    num_train_epochs: int = 2

    # train/valid split
    data_split: str = "0.98,0.02"
    max_train_samples: Union[None, int] = None
    max_valid_samples: Union[None, int] = None

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1

    seed: int = 42

    seq_length: int = 4096
    log_interval: int = 10
    step_checkpointing: bool = False
    checkpointing_steps: int = 100

    step_evaluation: bool = False
    evaluation_steps: int = 100

    # max train steps, if None, depends on num_train_epochs
    max_train_steps: Union[None, int] = None

    # if checkpointing every epoch, maybe True in sst
    epoch_checkpointing: bool = False
    epoch_evaluation: bool = False

    early_stopping: bool = False
    early_stopping_stall_num: int = 5

    attn_implementation: str = "flash_attention_2"

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


def prepare_args(args_type="Train"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, default=None)
    parsed = parser.parse_args()
    with open(parsed.c, 'r') as f:
        c = json.load(f)
    if args_type == "Train":
        args = TrainArgs(**c)
    else:
        raise ValueError("args_type must be Train")
    if not torch.cuda.is_available():
        args.attn_implementation = 'eager'

    return args
