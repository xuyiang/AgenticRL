from minimal_agent import fake_policy, run_agent, verify_answer


def test_fake_agent_completes_calculation() -> None:
    answer, trajectory = run_agent(
        task="计算 17 * 23",
        policy=fake_policy,
    )

    reward = verify_answer(answer, "391")

    assert answer == "391"
    assert reward == 1.0
    assert len(trajectory) == 2

    assert trajectory[0].action == "calculator"
    assert trajectory[0].tool_result is not None
    assert trajectory[0].tool_result.output == "391"

    assert trajectory[1].action == "final"
    assert trajectory[1].tool_result is None