def _format_float(num: float) -> str:
    return f"{num:.2f}".rstrip("0").rstrip(".")


def prompt_round_payoffs_base(
    chosen_action: int,
    round_payoff: float,
    num_volunteers: int,
    agent_name: str,
) -> str:
    """Post-decision narrative shown to one agent after the round resolves."""
    if chosen_action == 1:
        result = f"You, {agent_name}, chose Option 1 (Volunteer)."
    else:
        result = f"You, {agent_name}, chose Option 2 (Shirk)."

    if num_volunteers == 0:
        result += " Nobody volunteered."
    elif num_volunteers == 1:
        result += " One person volunteered."
    else:
        result += f" {num_volunteers} people volunteered."

    result += f" You received a payoff of {_format_float(round_payoff)}."
    return result


def prompt_opponent_action_report_base(
    agents: list[str],
    agent_id_to_name: dict[str, str],
    chosen_actions: dict[str, int],
) -> str:
    report = "Action report:"
    for agent in agents:
        if chosen_actions[agent] == 1:
            report += f" {agent_id_to_name[agent]} chose Option 1 (Volunteer)."
        else:
            report += f" {agent_id_to_name[agent]} chose Option 2 (Shirk)."
    return report
