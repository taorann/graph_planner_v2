import pytest

from graph_planner.agents.common.contracts import (
    ProtocolError,
    parse_action_block,
    validate_planner_action,
)
from graph_planner.core.actions import (
    ExploreAction,
    MemoryAction,
    NoopAction,
    RepairAction,
    SubmitAction,
)
def test_parse_and_validate_explore_success() -> None:
    text = (
        "<function=explore>\n"
        "  <param name=\"thought\">expand search</param>\n"
        "  <param name=\"op\"><![CDATA[expand]]></param>\n"
        "</function>"
    )
    parsed = parse_action_block(text)
    action = validate_planner_action(parsed)
    assert isinstance(action, ExploreAction)
    assert action.op == "expand"


def test_parse_and_validate_memory_success() -> None:
    text = (
        "<function=memory>\n"
        "  <param name=\"thought\">remember</param>\n"
        "  <param name=\"target\"><![CDATA[explore]]></param>\n"
        "  <param name=\"intent\"><![CDATA[commit]]></param>\n"
        "</function>"
    )
    parsed = parse_action_block(text)
    action = validate_planner_action(parsed)
    assert isinstance(action, MemoryAction)
    assert action.intent == "commit"


def test_parse_and_validate_repair_success() -> None:
    text = (
        "<function=repair>\n"
        "  <param name=\"thought\">fix bug</param>\n"
        "  <param name=\"subplan\"><![CDATA[1) change code]]></param>\n"
        "  <param name=\"focus_ids\">[\"n1\"]</param>\n"
        "  <param name=\"apply\">true</param>\n"
        "</function>"
    )
    parsed = parse_action_block(text)
    action = validate_planner_action(parsed)
    assert isinstance(action, RepairAction)
    assert action.plan == "1) change code"
    assert action.apply is True


def test_parse_and_validate_submit_success() -> None:
    text = (
        "<function=submit>\n"
        "  <param name=\"thought\">done</param>\n"
        "</function>"
    )
    parsed = parse_action_block(text)
    action = validate_planner_action(parsed)
    assert isinstance(action, SubmitAction)


def test_parse_and_validate_noop_success() -> None:
    text = "<function=noop>\n  <param name=\"thought\">skip</param>\n</function>"
    parsed = parse_action_block(text)
    action = validate_planner_action(parsed)
    assert isinstance(action, NoopAction)


def test_parse_rejects_multiple_blocks() -> None:
    text = "<function=noop></function><function=noop></function>"
    with pytest.raises(ProtocolError) as excinfo:
        parse_action_block(text)
    assert excinfo.value.code == "invalid-multi-block"


def test_parse_rejects_extra_text() -> None:
    text = "noise <function=noop></function>"
    with pytest.raises(ProtocolError) as excinfo:
        parse_action_block(text)
    assert excinfo.value.code == "extra-text"


def test_parse_rejects_unknown_action() -> None:
    text = "<function=unknown></function>"
    with pytest.raises(ProtocolError) as excinfo:
        parse_action_block(text)
    assert excinfo.value.code == "unknown-action"


def test_parse_rejects_duplicate_param() -> None:
    text = (
        "<function=memory>\n"
        "  <param name=\"thought\">remember</param>\n"
        "  <param name=\"target\"><![CDATA[explore]]></param>\n"
        "  <param name=\"target\"><![CDATA[observation]]></param>\n"
        "</function>"
    )
    with pytest.raises(ProtocolError) as excinfo:
        parse_action_block(text)
    assert excinfo.value.code == "duplicate-param"


def test_parse_rejects_forbidden_param() -> None:
    text = (
        "<function=noop>\n"
        "  <param name=\"thought\">skip</param>\n"
        "  <param name=\"target\">\"explore\"</param>\n"
        "</function>"
    )
    with pytest.raises(ProtocolError) as excinfo:
        parse_action_block(text)
    assert excinfo.value.code == "unknown-param"


def test_validate_requires_subplan_for_repair() -> None:
    text = "<function=repair>\n  <param name=\"thought\">plan</param>\n</function>"
    parsed = parse_action_block(text)
    with pytest.raises(ProtocolError) as excinfo:
        validate_planner_action(parsed)
    assert excinfo.value.code == "missing-required-param"


def test_parse_rejects_invalid_json() -> None:
    text = (
        "<function=repair>\n"
        "  <param name=\"thought\">plan</param>\n"
        "  <param name=\"subplan\"><![CDATA[step]]></param>\n"
        "  <param name=\"focus_ids\">[invalid</param>\n"
        "</function>"
    )
    with pytest.raises(ProtocolError) as excinfo:
        parse_action_block(text)
    assert excinfo.value.code == "invalid-json-param"
