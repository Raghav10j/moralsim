import os
from typing import List

from omegaconf import DictConfig

from moralsim.persona import EmbeddingModel
from moralsim.utils import ModelWandbWrapper

from .environment import VolunteerPerturbationEnv
from .persona import VolunteerPersona, VolunteerDummyPersona
from .persona.cognition import utils as cognition_utils
from ..common.persona.cognition import utils as common_cognition_utils
from ..common.run_utils import init_all_personas, run_step


def run(
    cfg: DictConfig,
    wandb_logger: ModelWandbWrapper,
    wrappers: List[ModelWandbWrapper],
    framework_wrapper: ModelWandbWrapper,
    embedding_model: EmbeddingModel,
    experiment_storage: str,
    seed: int,
):
    cognition_utils.SYS_VERSION = cfg.agent.system_prompt
    cognition_utils.KNOWLEDGE_CONDITION = cfg.agent.knowledge_condition

    if cfg.agent.cot_prompt == "think_step_by_step":
        common_cognition_utils.REASONING = "think_step_by_step"
    elif cfg.agent.cot_prompt == "deep_breath":
        common_cognition_utils.REASONING = "deep_breath"

    cognition_utils.OTHER_PERSONAS = [
        cfg.personas[f"persona_{i}"].name for i in range(cfg.personas.num)
    ]

    num_personas = cfg.personas.num

    personas = {
        f"persona_{i}": VolunteerPersona(
            cfg.agent,
            wrappers[i],
            framework_wrapper,
            embedding_model,
            os.path.join(experiment_storage, f"persona_{i}"),
            cfg.scenario,
        ) if (actions := cfg.personas[f"persona_{i}"].actions) is None
        else VolunteerDummyPersona(cfg.agent, wrappers[i], actions, cfg.env.max_num_rounds)
        for i in range(num_personas)
    }

    agent_name_to_id, agent_id_to_name = init_all_personas(personas, num_personas, cfg)

    env = VolunteerPerturbationEnv(cfg.env, experiment_storage, agent_id_to_name, seed=seed)

    agent_id, obs = env.reset(seed=seed)
    has_next_step = True
    while has_next_step:
        agent = personas[agent_id]
        has_next_step, agent_id, obs = run_step(env, agent, obs, num_personas, wandb_logger)

        if has_next_step:
            wandb_logger.save(experiment_storage, agent_name_to_id)

    env.save_log()
    for persona in personas:
        if isinstance(personas[persona], VolunteerPersona):
            personas[persona].memory.save()
