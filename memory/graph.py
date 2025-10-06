import ast
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple, Iterable

class RepoGraph:
    """
    轻量级文件图：
    - 节点：Python 文件路径（相对仓库根）
    - 边：import 无向近似 + 同目录弱边
    """
    def __init__(self, root: str):
        self.root = Path(root)
        self.adj: Dict[str, Set[str]] = defaultdict(set)
        self._index: Dict[str, str] = {}  # module_name -> file path (rel)

    def build(self, glob="**/*.py"):
        for p in self.root.rglob(glob):
            if not p.is_file(): continue
            rel = p.relative_to(self.root).as_posix()
            mod = rel[:-3].replace("/", ".")
            self._index[mod] = rel

        for p in self.root.rglob(glob):
            if not p.is_file(): continue
            rel = p.relative_to(self.root).as_posix()
            try:
                tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            imports = []
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    imports.extend(a.name for a in n.names)
                elif isinstance(n, ast.ImportFrom):
                    if n.module: imports.append(n.module)
            for mod in imports:
                parts = mod.split(".")
                for k in range(len(parts), 0, -1):
                    cand = ".".join(parts[:k])
                    if cand in self._index:
                        tgt = self._index[cand]
                        if tgt != rel:
                            self.adj[rel].add(tgt)
                            self.adj[tgt].add(rel)
                        break
            # 同目录弱边
            for sib in p.parent.glob("*.py"):
                r2 = sib.relative_to(self.root).as_posix()
                if r2 != rel:
                    self.adj[rel].add(r2)
                    self.adj[r2].add(rel)

    def expand(self, seeds: Iterable[str], max_hop:int=2) -> List[Tuple[str,int]]:
        seeds = [s for s in seeds if s in self.adj]
        dist: Dict[str,int] = {}
        q = deque()
        for s in seeds:
            dist[s] = 0
            q.append(s)
        while q:
            u = q.popleft()
            if dist[u] >= max_hop: 
                continue
            for v in self.adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return [(p,h) for p,h in dist.items() if h>0 and h<=max_hop]
