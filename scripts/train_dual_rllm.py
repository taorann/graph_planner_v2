# scripts/train_dual_rllm.py
import argparse
import importlib
import os
from typing import Any, Dict, Tuple, List

import ray
from omegaconf import OmegaConf
from importlib.resources import files

# ---------- helpers ----------

def _load_yaml(path: str):
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)  # allow new keys
    return cfg

def _load_rllm_default() -> "OmegaConf":
    # 尽量复用 rLLM/verl 的默认 trainer 配置
    for pkg in ("rllm.trainer.config", "rllm.train.config"):
        try:
            base = files(pkg)
            for name in ("agent_ppo_trainer.yaml", "_generated_agent_ppo_trainer.yaml", "ppo_trainer.yaml"):
                p = base.joinpath(name)
                try:
                    return OmegaConf.load(p)
                except Exception:
                    pass
        except Exception:
            continue
    raise RuntimeError("Cannot locate rLLM default PPO/GRPO config in installed package")

def _from_dotlist(dot: List[str]):
    return OmegaConf.from_dotlist(dot) if dot else OmegaConf.create({})

def _merge(*nodes: "OmegaConf") -> "OmegaConf":
    out = OmegaConf.create({})
    for n in nodes:
        if n is None: 
            continue
        out = OmegaConf.merge(out, n)
    return out

def _import_obj(path: str):
    # "a.b.c:ClassName" or "a.b.c.ClassName"
    if ":" in path:
        mod, name = path.split(":")
    else:
        parts = path.split(".")
        mod, name = ".".join(parts[:-1]), parts[-1]
    return getattr(importlib.import_module(mod), name)

def _role_cfg(base_cfg: "OmegaConf", role: str) -> "OmegaConf":
    """
    将 test4g.yaml 中的 role 子树（planner/cgm）映射到 rLLM trainer 所需扁平键：
    data / actor_rollout_ref / agent / trainer / ray_init / reward_model ...
    """
    role_node = OmegaConf.select(base_cfg, role, default=OmegaConf.create({}))
    common = OmegaConf.select(base_cfg, "common", default=OmegaConf.create({}))

    # 优先级：rLLM默认  <- common  <- 角色(role)  <- 角色.role_overrides
    default = _load_rllm_default()
    cfg = _merge(default, common, role_node, role_node.get("role_overrides", {}))

    # 兜底：开启 GRPO
    if OmegaConf.select(cfg, "actor_rollout_ref.actor.use_kl_loss") is None:
        cfg.actor_rollout_ref.actor.use_kl_loss = True
    if OmegaConf.select(cfg, "actor_rollout_ref.actor.kl_loss_type") is None:
        cfg.actor_rollout_ref.actor.kl_loss_type = "forward_kl"
    if OmegaConf.select(cfg, "actor_rollout_ref.actor.kl_loss_coef") is None:
        cfg.actor_rollout_ref.actor.kl_loss_coef = 0.02

    # 必要位若缺失，给出友好提示（不强退）
    must = [
        "actor_rollout_ref.model.path",
        "actor_rollout_ref.actor.optim.total_training_steps",
        "data.train_files", "data.val_files",
        "data.max_prompt_length", "data.max_response_length",
    ]
    missing = [k for k in must if OmegaConf.select(cfg, k) in (None, -1)]
    if missing:
        print(f"[{role.upper()}][WARN] missing keys:", missing)

    return cfg

# ---------- ray remote training ----------

@ray.remote
def run_trainer_remote(cfg_dict: Dict[str, Any], agent_cls_path: str, env_cls_path: str):
    from rllm.trainer.agent_trainer import AgentTrainer
    cfg = OmegaConf.create(cfg_dict)
    agent_cls = _import_obj(agent_cls_path)
    env_cls = _import_obj(env_cls_path)
    trainer = AgentTrainer(agent_class=agent_cls, env_class=env_cls, config=cfg)
    trainer.train()
    return "done"

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/experiments/test4g.yaml")
    ap.add_argument("--overrides", nargs="*", default=[], help="dot-list overrides k=v")
    ap.add_argument("--no-train-cgm", action="store_true", help="只训练 planner")
    args, unknown = ap.parse_known_args()

    base = _merge(_load_yaml(args.config), _from_dotlist(args.overrides + unknown))

    # 读取类路径（放在 test4g.yaml: common.agent_class / common.env_class，或各角色覆盖）
    planner_agent_cls = OmegaConf.select(base, "planner.agent_class") or OmegaConf.select(base, "common.agent_class")
    cgm_agent_cls     = OmegaConf.select(base, "cgm.agent_class")     or OmegaConf.select(base, "common.agent_class")
    env_cls           = OmegaConf.select(base, "planner.env_class")   or OmegaConf.select(base, "common.env_class")
    assert planner_agent_cls and env_cls, "agent_class/env_class 必须在 test4g.yaml 里给出"

    # 构造两个角色的最终 cfg
    planner_cfg = _role_cfg(base, "planner")
    cgm_cfg     = _role_cfg(base, "cgm")

    # 资源分配（默认 2+2，可在 test4g.yaml 里各自配置 role.resource.num_gpus）
    p_gpus = OmegaConf.select(base, "planner.resource.num_gpus", default=2)
    c_gpus = OmegaConf.select(base, "cgm.resource.num_gpus",     default=2)

    # Ray 启动
    ray_init = OmegaConf.select(base, "common.ray_init", default={})
    print("[RAY] init with:", ray_init)
    ray.init(**ray_init)

    # 提交 planner 训练
    tasks = []
    print("[PLANNER] launching...")
    tasks.append(
        run_trainer_remote.options(num_gpus=float(p_gpus)).remote(
            OmegaConf.to_container(planner_cfg, resolve=True),
            planner_agent_cls,
            env_cls,
        )
    )

    # 可选：提交 CGM 训练（默认开启；--no-train-cgm 可关闭）
    if not args.no_train_cgm:
        assert cgm_agent_cls, "cgm.agent_class 未配置（或用 --no-train-cgm 关闭）"
        print("[CGM] launching...")
        tasks.append(
            run_trainer_remote.options(num_gpus=float(c_gpus)).remote(
                OmegaConf.to_container(cgm_cfg, resolve=True),
                cgm_agent_cls,
                env_cls,
            )
        )

    # 阻塞等待
    ray.get(tasks)
    print("[OK] training finished.")

if __name__ == "__main__":
    main()
