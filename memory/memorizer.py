import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
from aci.runner import execute
from aci.hunkmap import LineMapper
from .types import Plan, Target, Subgraph, SubgraphNode, Feedback
from .graph import RepoGraph
from .selector import Selector, RuleSelector, Evidence

SYM_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
FILE_RE = re.compile(r"[A-Za-z0-9_/.-]+\.py")

@dataclass
class MemorizerConfig:
    K: int = 6
    W: int = 12
    MAX_HITS: int = 30
    CONTEXT: int = 2
    MAX_STEPS: int = 12
    MAX_LINES_PER_EDIT: int = 6
    EXPAND_HOPS: int = 2
    DIVERSITY_PER_DIR: int = 3
    STRUCT_EDGE_W: float = 1.0
    TELEPORT_W: float = 0.5

class Memorizer:
    def __init__(self, repo_root: str, cfg: MemorizerConfig = MemorizerConfig(),
                 selector: Optional[Selector]=None):
        self.repo = repo_root
        self.cfg = cfg
        self.mapper = LineMapper()
        self.S = Subgraph()
        self._issue_text = ""
        self.selector = selector or RuleSelector()
        # 文件级图
        self.graph = RepoGraph(repo_root)
        self.graph.build()

    # ---------- BOOTSTRAP ----------
    def bootstrap(self, issue_text: str) -> Subgraph:
        self._issue_text = issue_text
        symbols = self._extract_symbols(issue_text)
        paths   = self._extract_paths(issue_text)
        queries = list(dict.fromkeys(paths + symbols))

        hits = []
        for q in queries:
            resp = execute({"tool":"search","args":{
                "root": self.repo, "query": re.escape(q), "glob":"**/*.py",
                "max_hits": self.cfg.MAX_HITS, "context": self.cfg.CONTEXT}})
            for h in resp.get("observations",{}).get("search_hits",[]):
                score = 1.0 + (0.5 if (q in paths and h["path"].endswith(q)) else 0.0)
                hits.append(SubgraphNode(kind="HIT", path=h["path"], line=h["line"], score=score,
                                         window=self._mk_window(h["line"])))
        self.S = Subgraph(nodes=self._select_topK_diverse(hits, self.cfg.K, self.cfg.DIVERSITY_PER_DIR))
        return self.S

    # ---------- PLAN ----------
    def plan(self, failing_test_hint: str = "") -> Plan:
        nodes = sorted(self.S.nodes, key=lambda x: x.score, reverse=True)[:self.cfg.K]
        ev = Evidence(issue=self._issue_text, first_failure_frame=None, top_assert=failing_test_hint)
        targets = self.selector.select(
            candidates=nodes, K=self.cfg.K, per_dir=self.cfg.DIVERSITY_PER_DIR,
            default_window_fn=self._mk_window, evidence=ev
        )
        budget = {"max_steps": self.cfg.MAX_STEPS,
                  "max_lines_per_edit": self.cfg.MAX_LINES_PER_EDIT,
                  "lint_required": True}
        return Plan(targets=targets, budget=budget,
                    priority_tests=[failing_test_hint] if failing_test_hint else [])

    # ---------- UPDATE ----------
    def update(self, fb: Feedback) -> None:
        # 1) 行号映射
        if fb.diff_summary:
            path = self._guess_path_from_diff(fb.diff_summary)
            if path:
                self.mapper.apply_diff(path, fb.diff_summary)

        # 2) 以失败帧 + 当前子图文件为种子
        seeds = set()
        if fb.first_failure_frame and fb.first_failure_frame.get("path"):
            seeds.add(self._relpath(fb.first_failure_frame["path"]))
        for n in self.S.nodes:
            seeds.add(n.path)

        # 3) RepoGraph 结构化扩展（1–2 跳）
        struct_cands = []
        if seeds:
            for path, hop in self.graph.expand(seeds, max_hop=self.cfg.EXPAND_HOPS):
                score = self.cfg.STRUCT_EDGE_W * (0.7 ** (hop-1))
                center = 1
                struct_cands.append(SubgraphNode(kind="STRUCT", path=path, line=center,
                                                 score=score, window=self._mk_window(center)))

        # 4) 传送边：断言符号 IR 命中
        tele_cands = []
        for s in self._extract_symbols(fb.top_assert or "")[:8]:
            resp = execute({"tool":"search","args":{
                "root": self.repo, "query": re.escape(s), "glob":"**/*.py",
                "max_hits": min(10, self.cfg.MAX_HITS), "context": self.cfg.CONTEXT}})
            for h in resp.get("observations",{}).get("search_hits",[]):
                tele_cands.append(SubgraphNode(kind="TP", path=h["path"], line=h["line"],
                                               score=self.cfg.TELEPORT_W, window=self._mk_window(h["line"])))

        # 5) 合并候选 + 重新选择（目录多样性）
        pool = self.S.nodes + struct_cands + tele_cands
        ev = Evidence(issue=self._issue_text,
                      first_failure_frame=fb.first_failure_frame, top_assert=fb.top_assert)
        chosen = self.selector.select(pool, self.cfg.K, self.cfg.DIVERSITY_PER_DIR,
                                      self._mk_window, ev)

        # 6) 写回子图 + 行号重映射
        self.S.nodes = [SubgraphNode(kind="SEL", path=t.path, line=t.start, score=t.confidence,
                                     window=[t.start,t.end], meta={"why":t.why}) for t in chosen]
        for node in self.S.nodes:
            m = self.mapper.map_line(node.path, node.line)
            if m is not None:
                node.line = m
                node.window = self._mk_window(m)

    # ---------- helpers ----------
    def _select_topK_diverse(self, nodes: List[SubgraphNode], K:int, per_dir:int):
        by_key = {}
        for n in nodes:
            key = (n.path, n.line)
            if key not in by_key or by_key[key].score < n.score:
                by_key[key] = n
        nodes = list(by_key.values())
        nodes.sort(key=lambda x: x.score, reverse=True)

        buckets: Dict[str, int] = {}
        out: List[SubgraphNode] = []
        for n in nodes:
            if len(out) >= K: break
            d = "/".join(n.path.split("/")[:-1])
            if buckets.get(d,0) >= per_dir:
                continue
            out.append(n)
            buckets[d] = buckets.get(d,0) + 1
        return out

    def _mk_window(self, center: int):
        half = max(2, self.cfg.W//2)
        return [max(1, center-half), max(center+half, center)]

    @staticmethod
    def _extract_symbols(text: str) -> List[str]:
        cands = SYM_RE.findall(text or "")
        return [t for t in cands if not t.isdigit() and len(t) >= 3][:20]

    @staticmethod
    def _extract_paths(text: str) -> List[str]:
        return FILE_RE.findall(text or "")[:10]

    @staticmethod
    def _guess_path_from_diff(diff: str) -> str:
        m = re.search(r"^\+\+\+\s+(.+)$", diff, re.M)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _relpath(path: str) -> str:
        return Path(path).as_posix()
