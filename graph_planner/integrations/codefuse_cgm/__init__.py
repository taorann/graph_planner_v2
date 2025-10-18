"""Integration helpers for invoking CodeFuse CGM patching services."""

from importlib import import_module

from .client import CodeFuseCGMClient, build_cgm_payload

__all__ = [
    "CodeFuseCGMClient",
    "build_cgm_payload",
    "CGMExample",
    "CodeFuseCGMDataset",
    "GraphLinearizer",
    "SnippetFormatter",
    "ConversationEncoder",
    "CGMGenerationConfig",
    "CodeFuseCGMGenerator",
    "CGMBatchCollator",
    "CGMTrainingConfig",
    "CodeFuseCGMTrainer",
]


_LAZY_MODULES = {
    "CGMExample": "graph_planner.integrations.codefuse_cgm.data",
    "CodeFuseCGMDataset": "graph_planner.integrations.codefuse_cgm.data",
    "GraphLinearizer": "graph_planner.integrations.codefuse_cgm.data",
    "SnippetFormatter": "graph_planner.integrations.codefuse_cgm.data",
    "ConversationEncoder": "graph_planner.integrations.codefuse_cgm.formatting",
    "CGMGenerationConfig": "graph_planner.integrations.codefuse_cgm.inference",
    "CodeFuseCGMGenerator": "graph_planner.integrations.codefuse_cgm.inference",
    "CGMBatchCollator": "graph_planner.integrations.codefuse_cgm.training",
    "CGMTrainingConfig": "graph_planner.integrations.codefuse_cgm.training",
    "CodeFuseCGMTrainer": "graph_planner.integrations.codefuse_cgm.training",
}


def __getattr__(name: str):  # pragma: no cover - exercised during import
    module_name = _LAZY_MODULES.get(name)
    if module_name:
        module = import_module(module_name)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
