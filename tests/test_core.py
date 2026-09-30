"""Check the contracts shared by data, rewards and branching without model weights."""

import pytest
from tooltune.protocol import messages_for, parse_action
from tooltune.rewards import group_rewards, efficiency_weight
from tooltune.rollout import Rollout, PathState
from tooltune.tools.search import LocalDocument, LocalSearch
from tooltune.verifier import verify


def candidate(i, calls, correct=True, group="q"):
    return dict(
        group_id=group,
        candidate_id=i,
        logical_calls=calls,
        correct=correct,
        invalid_calls=0,
    )


def test_correctness_gates_cost_and_reference_is_cheapest_correct():
    rows = [
        candidate(0, 2),
        candidate(1, 5),
        candidate(2, 0, False),
        candidate(3, 1, False),
    ]
    rewards = group_rewards(rows, lam=0.1)
    assert rewards == pytest.approx([1, 0.94, 0, 0])


def test_all_wrong_group_gets_no_cost_bonus():
    assert group_rewards([candidate(i, i, False) for i in range(4)]) == [0] * 4


def test_cost_schedule():
    assert efficiency_weight(0, 1000) == 0
    assert efficiency_weight(200, 1000) == 0
    assert efficiency_weight(600, 1000) == pytest.approx(0.05)
    assert efficiency_weight(1000, 1000) == pytest.approx(0.1)


def test_groups_fail_closed():
    with pytest.raises(ValueError):
        group_rewards([candidate(0, 1)])
    with pytest.raises(ValueError):
        group_rewards([candidate(0, 1)] * 4)
    with pytest.raises(ValueError):
        group_rewards([candidate(i, 1, group=str(i % 2)) for i in range(4)])
    with pytest.raises(RuntimeError):
        group_rewards(
            [dict(candidate(i, 1), infrastructure_error=True) for i in range(4)]
        )


def test_task_reference_is_not_exposed_to_model():
    task = {
        "question": "Find the value.",
        "answer": "SECRET_GOLD",
        "documents": [{"text": "HIDDEN_DOCUMENT"}],
    }
    prompt = str(messages_for(task))
    assert "SECRET_GOLD" not in prompt and "HIDDEN_DOCUMENT" not in prompt


def test_ambiguous_protocol_is_rejected():
    call = '<tool_call>{"name":"python","arguments":{"code":"print(2)"}}</tool_call>'
    assert parse_action(call)[0] == "python"
    assert parse_action(call + "<final_answer>2</final_answer>")[0] == "invalid"
    assert parse_action(call + call)[0] == "invalid"
    assert parse_action("<final_answer>2</final_answer>") == ("final", "2")


@pytest.mark.parametrize(
    "prediction,expected",
    [("1,000", True), ("1000 or 2", False), ("1000, 2", False), ("2000/2", True)],
)
def test_whole_final_numeric_verification(prediction, expected):
    assert verify(prediction, "1000", "numeric") is expected


def test_local_search_returns_only_supplied_documents():
    search = LocalSearch(
        [
            LocalDocument("a", "Luma holds 27.", "Luma"),
            LocalDocument("b", "Vex holds 41.", "Vex"),
        ]
    )
    result = search.search("Luma")
    assert "27" in result and "41" in result
    assert search.search("") == "SEARCH_ERROR: empty query"


class ScriptedRollout(Rollout):
    """A fixture emits root paths; the production branch-selection code is exercised."""

    def new(self, task, group, candidate, efficient=False):
        return PathState(task, group, candidate, candidate, [1], messages=[])

    def complete(self, states):
        for state in states:
            if state.branch_point is None:
                state.ids = [11, 12, 90, 21, 22]
                state.mask = [1, 1, 0, 1, 1]
                state.logprobs = [-0.1, -0.2, 0.0, -0.3, -0.4]
                state.logical_calls = 1
                state.physical_calls = 1
                state.events = [{"tool": "search", "args": {"query": "x"}}]
                state.observations = [
                    dict(
                        offset=3,
                        tool="search",
                        event_count=1,
                        message_count=0,
                        calls=1,
                        invalid=0,
                        model_start=0,
                    )
                ]
            else:
                state.ids += [31, 32]
                state.mask += [1, 1]
                state.logprobs += [-0.5, -0.6]
            state.stop = "final"
            state.final_answer = "2"
        return states

    def entropy(self, model, state, observation):
        return 0.3


def test_branching_preserves_group_size_prefix_and_logical_cost():
    engine = ScriptedRollout(None, None)
    engine.history["global"].extend([0.1, 0.2])
    try:
        rows = engine.sample([{"task_id": "q"}], mode="entropy")
        assert len(rows) == 4
        assert len({r.group_id for r in rows}) == 1
        for child in rows[2:]:
            assert child.branch_point == 3
            assert child.ids == [11, 12, 90, 31, 32]
            assert child.mask == [1, 1, 0, 1, 1]
            assert child.logical_calls == 1 and child.physical_calls == 0
        assert rows[0].ids == [11, 12, 90, 21, 22]
    finally:
        engine.pool.shutdown()


def test_no_qualifying_branch_fills_with_independent_roots():
    engine = ScriptedRollout(None, None)
    engine.history["global"].extend([0.8, 0.9])
    try:
        rows = engine.sample([{"task_id": "q"}], mode="entropy")
        assert len(rows) == 4 and all(r.branch_point is None for r in rows)
    finally:
        engine.pool.shutdown()


def test_tool_histories_use_global_fallback_then_separate_threshold():
    engine = ScriptedRollout(None, None)
    engine.history["global"].extend([0.1] * 32)
    engine.history["search"].extend([0.8] * 31)
    try:
        rows = engine.sample([{"task_id": "q"}], mode="calibrated")
        assert sum(r.branch_point is not None for r in rows) == 2
        engine.history["search"].clear()
        engine.history["search"].extend([0.8] * 32)
        rows = engine.sample([{"task_id": "q"}], mode="calibrated")
        assert all(r.branch_point is None for r in rows)
    finally:
        engine.pool.shutdown()
