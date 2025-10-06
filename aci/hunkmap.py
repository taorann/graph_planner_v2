from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import difflib, re

@dataclass
class Hunk:
    old_start: int
    old_len:   int
    new_start: int
    new_len:   int

HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

def compute_unified_diff(old:str, new:str, path:str, n_context:int=3)->str:
    old_lines = old.splitlines(True)
    new_lines = new.splitlines(True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path, n=n_context)
    return "".join(diff)

def parse_unified_diff(diff_text: str) -> List[Hunk]:
    hunks: List[Hunk] = []
    for line in diff_text.splitlines():
        m = HUNK_RE.match(line)
        if not m: 
            continue
        o_s, o_l, n_s, n_l = m.groups()
        hunks.append(
            Hunk(
                old_start=int(o_s),
                old_len=int(o_l) if o_l else 1,
                new_start=int(n_s),
                new_len=int(n_l) if n_l else 1,
            )
        )
    return hunks

class LineMapper:
    """
    分段线性行号映射器：
    - hunk 之前：累积 offset 后映射
    - 落在 hunk 内：返回 None（不保证一一对应）
    - hunk 之后：继续累积 offset
    """
    def __init__(self) -> None:
        self._file_hunks: Dict[str, List[Hunk]] = {}

    def apply_diff(self, path: str, diff_text: str) -> None:
        hunks = parse_unified_diff(diff_text)
        if not hunks:
            return
        self._file_hunks.setdefault(path, []).extend(hunks)

    def map_line(self, path: str, old_line: int) -> Optional[int]:
        hunks = self._file_hunks.get(path, [])
        offset = 0
        for h in hunks:
            old_h_end = h.old_start + h.old_len - 1
            if old_line < h.old_start:
                return old_line + offset
            if h.old_start <= old_line <= old_h_end:
                return None
            offset += (h.new_len - h.old_len)
        return old_line + offset

    def map_span(self, path: str, start: int, end: int) -> Tuple[Optional[int], Optional[int]]:
        return self.map_line(path, start), self.map_line(path, end)
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import difflib, re

@dataclass
class Hunk:
    old_start: int
    old_len:   int
    new_start: int
    new_len:   int

HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

def compute_unified_diff(old:str, new:str, path:str, n_context:int=3)->str:
    old_lines = old.splitlines(True)
    new_lines = new.splitlines(True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path, n=n_context)
    return "".join(diff)

def parse_unified_diff(diff_text: str) -> List[Hunk]:
    hunks: List[Hunk] = []
    for line in diff_text.splitlines():
        m = HUNK_RE.match(line)
        if not m: 
            continue
        o_s, o_l, n_s, n_l = m.groups()
        hunks.append(
            Hunk(
                old_start=int(o_s),
                old_len=int(o_l) if o_l else 1,
                new_start=int(n_s),
                new_len=int(n_l) if n_l else 1,
            )
        )
    return hunks

class LineMapper:
    """
    分段线性行号映射器：
    - hunk 之前：累积 offset 后映射
    - 落在 hunk 内：返回 None（不保证一一对应）
    - hunk 之后：继续累积 offset
    """
    def __init__(self) -> None:
        self._file_hunks: Dict[str, List[Hunk]] = {}

    def apply_diff(self, path: str, diff_text: str) -> None:
        hunks = parse_unified_diff(diff_text)
        if not hunks:
            return
        self._file_hunks.setdefault(path, []).extend(hunks)

    def map_line(self, path: str, old_line: int) -> Optional[int]:
        hunks = self._file_hunks.get(path, [])
        offset = 0
        for h in hunks:
            old_h_end = h.old_start + h.old_len - 1
            if old_line < h.old_start:
                return old_line + offset
            if h.old_start <= old_line <= old_h_end:
                return None
            offset += (h.new_len - h.old_len)
        return old_line + offset

    def map_span(self, path: str, start: int, end: int) -> Tuple[Optional[int], Optional[int]]:
        return self.map_line(path, start), self.map_line(path, end)
