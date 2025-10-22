"""Toy MLP causal language model used for lightweight integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

__all__ = [
    "ToyLMConfig",
    "ToyLMForCausalLM",
    "ToyTokenizer",
    "create_toy_checkpoint",
]


def _default_vocab() -> List[str]:
    """Return a stable vocabulary covering printable ASCII characters."""

    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
    ascii_tokens = [chr(idx) for idx in range(32, 127)]
    return specials + ascii_tokens


class ToyTokenizer(PreTrainedTokenizer):
    """Character-level tokenizer compatible with Hugging Face's auto APIs."""

    vocab_files_names = {"vocab_file": "toy_vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        *,
        vocab: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> None:
        if vocab_file:
            tokens = json.loads(Path(vocab_file).read_text(encoding="utf-8"))
        else:
            tokens = list(vocab) if vocab is not None else _default_vocab()
        self._token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(tokens)}
        self._id_to_token: Dict[int, str] = {idx: tok for tok, idx in self._token_to_id.items()}

        kwargs.setdefault("bos_token", "<bos>")
        kwargs.setdefault("eos_token", "<eos>")
        kwargs.setdefault("unk_token", "<unk>")
        kwargs.setdefault("pad_token", "<pad>")
        super().__init__(**kwargs)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._token_to_id)

    @property
    def vocab_size(self) -> int:  # type: ignore[override]
        return len(self._token_to_id)

    def _tokenize(self, text: str, **kwargs) -> List[str]:  # type: ignore[override]
        del kwargs
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tok for tok in tokens if tok not in {self.bos_token, self.eos_token, self.pad_token})

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        output = [self.bos_token_id] + list(token_ids_0) + [self.eos_token_id]
        if token_ids_1 is not None:
            output += list(token_ids_1) + [self.eos_token_id]
        return output

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return [1 if token_id in {self.bos_token_id, self.eos_token_id, self.pad_token_id} else 0 for token_id in token_ids_0]
        mask = [1]
        mask.extend(0 for _ in token_ids_0)
        mask.append(1)
        if token_ids_1 is not None:
            mask.extend(0 for _ in token_ids_1)
            mask.append(1)
        return mask

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        length = len(token_ids_0) + 2
        if token_ids_1 is not None:
            length += len(token_ids_1) + 1
        return [0] * length

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        name = "toy_vocab.json" if filename_prefix is None else f"{filename_prefix}-toy_vocab.json"
        path = Path(save_directory) / name
        path.write_text(json.dumps([self._id_to_token[idx] for idx in range(len(self._id_to_token))]), encoding="utf-8")
        return (str(path),)


class ToyLMConfig(PretrainedConfig):
    """Configuration describing the toy causal LM architecture."""

    model_type = "toy_lm"

    def __init__(
        self,
        vocab_size: int = len(_default_vocab()),
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        **kwargs,
    ) -> None:
        kwargs.setdefault("pad_token_id", 0)
        kwargs.setdefault("bos_token_id", 1)
        kwargs.setdefault("eos_token_id", 2)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_cache = False


class ToyLMForCausalLM(PreTrainedModel, GenerationMixin):
    """Minimal feed-forward network with causal LM head."""

    config_class = ToyLMConfig

    def __init__(self, config: ToyLMConfig) -> None:  # type: ignore[override]
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        layers: List[nn.Module] = []
        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(config.hidden_size, config.hidden_size))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutput:
        del kwargs
        input_ids = input_ids.long()
        hidden = self.embed(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).type_as(hidden)
            hidden = hidden * mask
        hidden = self.mlp(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, torch.Tensor]:
        return {"input_ids": input_ids}

    def _reorder_cache(self, past, beam_idx):  # noqa: D401
        return past


def create_toy_checkpoint(save_directory: Path | str) -> Path:
    """Persist a toy model/tokenizer pair to ``save_directory``."""

    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ToyTokenizer()
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
    )
    tokenizer.save_pretrained(save_dir)

    config = ToyLMConfig(vocab_size=len(tokenizer.get_vocab()))
    config.auto_map = {
        "AutoConfig": "graph_planner.models.toy_lm.ToyLMConfig",
        "AutoModelForCausalLM": "graph_planner.models.toy_lm.ToyLMForCausalLM",
        "AutoTokenizer": "graph_planner.models.toy_lm.ToyTokenizer",
    }
    config.architectures = [ToyLMForCausalLM.__name__]
    config.save_pretrained(save_dir)

    model = ToyLMForCausalLM(config)
    model.save_pretrained(save_dir)
    return save_dir


AutoConfig.register(ToyLMConfig.model_type, ToyLMConfig)
AutoModelForCausalLM.register(ToyLMConfig, ToyLMForCausalLM)
AutoTokenizer.register(ToyLMConfig, slow_tokenizer_class=ToyTokenizer)
