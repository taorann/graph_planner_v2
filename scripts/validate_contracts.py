#!/usr/bin/env python3
"""Sanity-check the planner and CGM contracts against golden samples."""

from __future__ import annotations

from graph_planner.agents.common.contracts import (
    ProtocolError,
    parse_action_block,
    validate_cgm_patch,
    validate_planner_action,
)


def _sample_planner_block() -> str:
    return (
        "<function=repair>\n"
        "  <param name=\"thought\">plan repair</param>\n"
        "  <param name=\"subplan\"><![CDATA[1) change code]]></param>\n"
        "  <param name=\"focus_ids\">[\"n1\"]</param>\n"
        "</function>"
    )


def _sample_cgm_payload() -> dict:
    return {
        "patch": {
            "edits": [
                {"path": "app.py", "start": 1, "end": 1, "new_text": "print('patched')\n"},
            ]
        },
        "summary": "patch",
    }


def main() -> int:
    try:
        parsed = parse_action_block(_sample_planner_block())
        validate_planner_action(parsed)
        validate_cgm_patch(_sample_cgm_payload())
    except ProtocolError as exc:  # pragma: no cover - CLI feedback
        print(f"Validation failed: {exc.code}: {exc.detail}")
        return 1
    print("Contracts validated successfully.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
