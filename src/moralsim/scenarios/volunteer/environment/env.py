from omegaconf import DictConfig

from moralsim.scenarios.common.environment import MoralityPerturbationEnv

from .utils import (
    prompt_opponent_action_report_base,
    prompt_round_payoffs_base,
)

import logging

logger = logging.getLogger(__name__)


class VolunteerPerturbationEnv(MoralityPerturbationEnv):
    """One-shot 2-player Volunteer's Dilemma environment.

    Action encoding (mirrors PD):
        1 = VOLUNTEER, 2 = SHIRK

    Payoffs (constants, no per-round stochastic prize):
        - Volunteer:                                cfg.volunteer_payoff   (0.5)
        - Shirk when at least one other volunteers: cfg.free_rider_payoff  (1.0)
        - Shirk when nobody volunteers:             cfg.no_one_payoff      (0)
    """

    def __init__(
        self, cfg: DictConfig, experiment_storage: str, map_id_to_name: dict[str, str], seed: int,
    ) -> None:
        super().__init__(cfg, experiment_storage, map_id_to_name, seed)

    def _prompt_round_payoffs(self, agent: str) -> str:
        chosen_action = self.internal_global_state["chosen_action"][agent]
        round_payoff = self.internal_global_state["round_payoff"][agent]
        num_volunteers = self.internal_global_state["num_volunteers"]
        agent_name = self.agent_id_to_name[agent]
        if "vd_base" in self.cfg.name.lower():
            prompt_round_payoffs_fn = prompt_round_payoffs_base
        else:
            raise ValueError(f"Unexpected environment name: {self.cfg.name}.")
        return prompt_round_payoffs_fn(chosen_action, round_payoff, num_volunteers, agent_name)

    def _prompt_opponent_action_report(self) -> str:
        if "vd_base" in self.cfg.name.lower():
            prompt_opponent_action_report_fn = prompt_opponent_action_report_base
        else:
            raise ValueError(f"Unexpected environment name: {self.cfg.name}.")
        return prompt_opponent_action_report_fn(self.agents, self.agent_id_to_name, self.internal_global_state["chosen_action"])

    def _agent_no_survival(self, agent: str) -> bool:
        return False

    def _apply_perturbations(self) -> None:
        return

    def _generate_inputs_stochastic(self, **_) -> None:
        # VoD payoffs are constants; no stochastic prize. Stub keeps the base
        # env's _observe_pre_round happy (it reads last_collected_resource).
        for agent in self.agents:
            self.internal_global_state["last_collected_resource"][agent] = 0

    def _assign_resource(self) -> None:
        chosen_actions = self.internal_global_state["chosen_action"]
        num_volunteers = sum(1 for a in chosen_actions.values() if a == 1)
        self.internal_global_state["num_volunteers"] = num_volunteers

        for agent in self.agents:
            if chosen_actions[agent] == 1:
                res = self.cfg.volunteer_payoff
            elif num_volunteers > 0:
                res = self.cfg.free_rider_payoff
            else:
                res = self.cfg.no_one_payoff

            self.log_round(
                self.internal_global_state["action"][agent],
                self.internal_global_state["last_collected_resource"][agent],
                res,
            )

            self.internal_global_state["acc_payoff"][agent] += res
            self.internal_global_state["round_payoff"][agent] = res
            self.rewards[agent] += res
