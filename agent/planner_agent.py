# graph_planner/agent/planner_agent.py
from typing import Dict, Any
from core.actions import RepairAction, SubmitAction, ActionUnion

class PlannerAgent:
    """
    给执行引擎用的薄壳。后续换成 navigator.act(obs) 即可。
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _decide(self, obs: Dict[str, Any]) -> ActionUnion:
        if obs["steps"] == 0:
            return RepairAction(apply=False, issue=obs["issue"])
        return SubmitAction()

    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        act = self._decide(obs)
        return {
            "prompt": f"[ISSUE] {obs['issue'].get('id','NA')} [STEP] {obs['steps']}",
            "response": act.type,
            "action_obj": act,
        }
