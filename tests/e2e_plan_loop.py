# TODO: patch by CGM

if __name__ == "__main__":
    issue = "Fix failure in f1_score when y_true is None; see metrics/_classification.py"
    orch = Orchestrator(repo_root=".", use_llm_selector=False)
    result = orch.run_once(issue, max_rounds=3)
    print("OK =", result["ok"])
    print("Steps =", len(result["events"]))
    for ev in result["events"]:
        print("--- step", ev["step"], "summary:", ev["feedback"]["tests_failed"], "failed")
