from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from .types import SubgraphNode, Target

@dataclass
class Evidence:
    issue: str
    first_failure_frame: Optional[Dict[str,Any]] = None
    top_assert: Optional[str] = None

class Selector:
    def select(self, candidates: List[SubgraphNode], K:int, per_dir:int,
               default_window_fn: Callable[[int], List[int]],
               evidence: Evidence) -> List[Target]:
        raise NotImplementedError

class RuleSelector(Selector):
    """按score排序 + 目录多样性"""
    def select(self, candidates, K, per_dir, default_window_fn, evidence):
        buckets:{str,int} = {}
        uniq = {}
        for n in candidates:
            key = (n.path, n.line)
            if key not in uniq or uniq[key].score < n.score:
                uniq[key] = n
        nodes = sorted(uniq.values(), key=lambda x: x.score, reverse=True)
        out: List[Target] = []
        for n in nodes:
            if len(out) >= K: break
            d = "/".join(n.path.split("/")[:-1])
            if buckets.get(d,0) >= per_dir: continue
            w = n.window or default_window_fn(n.line)
            out.append(Target(path=n.path, start=w[0], end=w[1],
                              confidence=min(0.99, 0.5+0.1*n.score),
                              why=f"{n.kind.lower()}@{n.line}"))
            buckets[d] = buckets.get(d,0)+1
        return out

class LLMSelector(Selector):
    """可选：让 LLM 参与选择；失败自动回退到规则"""
    def __init__(self, call_model: Callable[[List[Dict[str,str]]], str]):
        self.call_model = call_model
        self.fallback = RuleSelector()

    def select(self, candidates, K, per_dir, default_window_fn, evidence):
        cand = [{
            "path": n.path, "line": n.line, "score": round(float(n.score),3),
            "kind": n.kind, "window": n.window or default_window_fn(n.line)
        } for n in candidates]
        messages = self._build_messages(cand, K, per_dir, evidence)
        try:
            raw = self.call_model(messages)
            data = self._safe_parse_json(raw)
            targets = []
            for t in data.get("targets", [])[:K]:
                targets.append(Target(
                    path=t["path"], start=int(t["start"]), end=int(t["end"]),
                    confidence=float(t.get("confidence", 0.7)),
                    why=t.get("why","llm_selected")))
            targets = self._enforce_diversity(targets, per_dir)
            return targets if targets else self.fallback.select(candidates,K,per_dir,default_window_fn,evidence)
        except Exception:
            return self.fallback.select(candidates,K,per_dir,default_window_fn,evidence)

    def _build_messages(self, cand, K, per_dir, ev: Evidence):
        sys = (f"你是代码修复的规划器。只输出JSON。最多选择 {K} 个目标，"
               f"同一目录最多 {per_dir} 个。每个目标是文件的行号窗口。")
        usr = {"issue": ev.issue, "first_failure_frame": ev.first_failure_frame,
               "top_assert": ev.top_assert, "candidates": cand,
               "output_schema":{"targets":"[{path,start,end,confidence,why}]"}}
        return [{"role":"system","content":sys},{"role":"user","content":str(usr)}]

    @staticmethod
    def _safe_parse_json(raw:str)->Dict[str,Any]:
        import json, re
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}

    @staticmethod
    def _enforce_diversity(targets: List[Target], per_dir:int)->List[Target]:
        out=[]; buckets={}
        for t in targets:
            d="/".join(t.path.split("/")[:-1])
            if buckets.get(d,0) >= per_dir: continue
            out.append(t); buckets[d]=buckets.get(d,0)+1
        return out
