from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ACIResponse:
    ok: bool
    summary: str
    effects: Dict[str, Any] = field(default_factory=dict)        # files_touched, lines_changed, diff_summary
    observations: Dict[str, Any] = field(default_factory=dict)   # snippets, search_hits, test_report
    limits: Dict[str, Any] = field(default_factory=dict)         # truncated, omitted_tokens
    metrics: Dict[str, Any] = field(default_factory=dict)        # elapsed_ms
