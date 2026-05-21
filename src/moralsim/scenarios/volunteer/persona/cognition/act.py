import logging
from datetime import datetime
from typing import Callable

from pathfinder import assistant, user

from moralsim.persona.cognition.act import ActComponent
from moralsim.persona.common import PersonaIdentity
from moralsim.scenarios.common.persona.cognition import MoralityActComponent
from moralsim.scenarios.common.persona.cognition.utils import reasoning_steps_prompt
from moralsim.utils import ModelWandbWrapper

logger = logging.getLogger(__name__)


class VolunteerDummyActComponent(ActComponent):
    def __init__(self, model: ModelWandbWrapper, cfg, actions: str | list, max_num_rounds: int):
        super().__init__(model, None)
        self.actions = self.compute_action_trajectory(actions, max_num_rounds)
        self.current_round = 0

    def compute_action_trajectory(self, actions: str | list, max_num_rounds: int):
        if isinstance(actions, list):
            if len(actions) != max_num_rounds:
                raise ValueError(
                    "List of actions was passed which doesn't match number of rounds."
                )
            return actions
        if actions == "volunteer":
            return [1] * max_num_rounds
        if actions == "shirk":
            return [2] * max_num_rounds
        raise ValueError(f"action strategy {actions} not supported.")

    def choose_action(self, context):
        res = self.actions[self.current_round]
        self.current_round += 1
        return res


class VolunteerActComponent(MoralityActComponent):
    def __init__(
        self,
        model: ModelWandbWrapper,
        model_framework: ModelWandbWrapper,
        cfg,
        system_prompt_fn: Callable[[PersonaIdentity], str],
        decision_prompt_fn: Callable[[str], str],
    ):
        super().__init__(model, model_framework, cfg, system_prompt_fn, decision_prompt_fn)

    def get_format_answer_prompt(self):
        return (
            'Respond with either 1 or 2 depending on the option number that corresponds to your decision'
            ' as your final answer after "Answer:". Example: "Answer: 1"'
        )

    def prompt_choose_action(
        self,
        model: ModelWandbWrapper,
        identity: PersonaIdentity,
        memories: list[str],
        current_location: str,
        current_time: datetime,
        context: str,
    ) -> tuple[float, str]:
        # Override of MoralityActComponent.prompt_choose_action: skip the
        # unconditional location_time_info and memory_prompt scaffolding
        # since one-shot VoD has no useful location/date or prior memory.
        lm = model.start_chain(
            identity.name, "cognition_act", "choose_act_options"
        )

        with user():
            lm += self.get_system_prompt(identity)
            lm += "\n"
            lm += self.get_decision_prompt(context)
            lm += "\n"
            lm += reasoning_steps_prompt()
            lm += self.get_format_answer_prompt()

        with assistant():
            lm = model.gen(
                lm,
                "reasoning",
                stop_regex=r"Answer:|So, the answer is:|he final answer is|\*\*Answer\*\*:",
                save_stop_text=True,
                max_tokens=8000,
            )
            lm = model.find(
                lm,
                regex=r"\d*\.?\d+",
                default_value="0",
                name="option",
            )
            option = float(lm["option"])

        model.end_chain(identity.name, lm)

        return option, lm.html()

    def choose_action(
        self,
        retrieved_memories: list[str],
        current_location: str,
        current_time: datetime,
        context: str,
    ) -> tuple[float, list[str]]:
        res, html = self.prompt_choose_action(
            self.model,
            self.persona.identity,
            retrieved_memories,
            current_location,
            current_time,
            context,
        )
        # Defensive clamp: MoralityActComponent.prompt_choose_action extracts
        # r"\d*\.?\d+", so a model that emits e.g. "Answer: 0.5" would return
        # a fractional value. Force the action onto the discrete {1, 2} space
        # and default unknowns to SHIRK so the env never crashes.
        try:
            option = int(round(float(res)))
        except (TypeError, ValueError):
            option = 2
        if option not in (1, 2):
            logger.warning(
                "VolunteerActComponent received non-binary option %s; defaulting to SHIRK (2).",
                res,
            )
            option = 2
        return float(option), [html]
