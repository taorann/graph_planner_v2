from __future__ import annotations

from pathlib import Path

import torch

from graph_planner.integrations.codefuse_cgm.data import CGMExample
from graph_planner.integrations.codefuse_cgm.inference import CGMGenerationConfig, CodeFuseCGMGenerator
from graph_planner.integrations.local_llm.hf import HuggingFaceChatClient, HuggingFaceChatConfig
from graph_planner.models import ToyLMConfig, ToyLMForCausalLM, create_toy_checkpoint


def _build_example() -> CGMExample:
    return CGMExample(
        prompt="Summarise the issue and propose a fix.",
        response="",
        graph={"nodes": [], "edges": []},
        plan="Apply toy fix",
        issue={"title": "Toy bug"},
        snippets=[{"path": "foo.py", "start": 1, "end": 1, "snippet": ["0001: pass"]}],
        metadata={},
    )


def test_toy_checkpoint_integrates_with_cgm_generator(tmp_path):
    checkpoint = create_toy_checkpoint(tmp_path / "toy")
    generator = CodeFuseCGMGenerator(
        CGMGenerationConfig(model_name_or_path=str(checkpoint), max_new_tokens=8, temperature=0.0)
    )

    outputs = generator.generate(_build_example())
    assert isinstance(outputs, list) and len(outputs) == 1
    assert isinstance(outputs[0], str)


def test_toy_checkpoint_integrates_with_planner_chat(tmp_path):
    checkpoint = create_toy_checkpoint(tmp_path / "toy_planner")
    client = HuggingFaceChatClient(
        HuggingFaceChatConfig(
            model_name_or_path=str(checkpoint),
            max_new_tokens=8,
            temperature=0.0,
            do_sample=False,
        )
    )

    response = client.chat([
        {"role": "system", "content": "You are a helpful planner."},
        {"role": "user", "content": "Plan a fix."},
    ])
    assert isinstance(response, str)


def test_toy_model_supports_backward_updates():
    config = ToyLMConfig(vocab_size=16, hidden_size=8, num_hidden_layers=1)
    model = ToyLMForCausalLM(config)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

    input_ids = torch.randint(low=0, high=config.vocab_size, size=(2, 6))
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)
    assert output.loss is not None

    before = model.lm_head.weight.detach().clone()

    output.loss.backward()
    optimiser.step()

    assert not torch.allclose(before, model.lm_head.weight)
