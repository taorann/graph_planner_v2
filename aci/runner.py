# aci/runner.py
import time, json
from .schema import ACIResponse
from .tools import view_file, search, edit_lines, run_tests, lint_check

DISPATCH = {
  "view_file": view_file,
  "search": search,
  "edit_lines": edit_lines,
  "run_tests": run_tests,
  "lint_check": lint_check,
}

def execute(req:dict)->dict:
    tool = req["tool"]; args = req.get("args", {})
    fn = DISPATCH.get(tool)
    if not fn:
        return ACIResponse(False, f"unknown tool: {tool}").__dict__
    try:
        res = fn(args)
        return res.__dict__
    except Exception as e:
        return ACIResponse(False, f"exception: {e}").__dict__
