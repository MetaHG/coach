from collections import OrderedDict
from typing import Union, List, Tuple

import numpy as np
from copy import deepcopy

from rl_coach.agents.dqn_agent import DQNNetworkParameters
from rl_coach.agents.ddqn_bcq_agent import DDQNBCQAlgorithmParameters, DDQNBCQAgent, DDQNBCQAgentParameters
from rl_coach.agents.bc_agent import BCNetworkParameters

from rl_coach.core_types import Batch
from rl_coach.off_policy_evaluators.custom_ope_manager import CustomOpeManager
from rl_coach.core_types import Transition, Episode
from rl_coach.architectures.architecture import Architecture
from rl_coach.logger import screen
from rl_coach.filters.filter import NoInputFilter

from rl_coach.architectures.head_parameters import PolicyHeadParameters
from rl_coach.architectures.head_parameters import QHeadParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.base_parameters import AgentParameters, AlgorithmParameters, NetworkParameters, MiddlewareScheme
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.memories.non_episodic.experience_replay import ExperienceReplayParameters

from rl_coach.schedules import LinearSchedule

class CustomBCAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()


class CustomBCAgentParameters(DDQNBCQAgentParameters):
    def __init__(self):
        super().__init__()
        self.network_wrappers['behavior_cloning_model'] = BCNetworkParameters()

    @property
    def path(self):
        return 'rl_coach.agents.custom_bc_agent:CustomBCAgent'

# class CustomBCAgentParameters(AgentParameters):
#     def __init__(self):
#         super().__init__(algorithm=DDQNBCQAlgorithmParameters(),
#                          exploration=EGreedyParameters(),
#                          memory=ExperienceReplayParameters(),
#                          networks={'main':DQNNetworkParameters(), 
#                                    'behavior_cloning_model': BCNetworkParameters()})
#         self.exploration.epsilon_schedule = LinearSchedule(1, 0.01, 1000000)
#         self.exploration.evaluation_epsilon = 0.001

#     @property
#     def path(self):
#         return 'rl_coach.agents.custom_bc_agent:CustomBCAgent'


class CustomBCAgent(DDQNBCQAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        
        self.ope_manager = CustomOpeManager()
        
    def get_behavior_cloning_model_loss(self, batch: Batch):
        network_keys = self.ap.network_wrappers['behavior_cloning_model'].input_embedders_parameters.keys()

        targets = np.ones(batch.actions().shape[0])

        return self.networks['behavior_cloning_model'].train_and_sync_networks(
            {**batch.states(network_keys), 'output_0_0': batch.actions()}, targets)[0]

    def improve_behavior_cloning_model(self, epochs: int):
        """
        Train a behavior cloning model to be used by the doubly-robust estimator
        :param epochs: The total number of epochs to use for training a reward model
        :return: None
        """
        batch_size = self.ap.network_wrappers['behavior_cloning_model'].batch_size

        # this is fitted from the training dataset
        for epoch in range(epochs):
            loss = 0
            total_transitions_processed = 0
            for i, batch in enumerate(self.call_memory('get_shuffled_training_data_generator', batch_size)):
                batch = Batch(batch)
                loss += self.get_behavior_cloning_model_loss(batch)
                total_transitions_processed += batch.size

            log = OrderedDict()
            log['Epoch'] = epoch
            log['loss'] = loss / total_transitions_processed
            screen.log_dict(log, prefix='Training Behavior Cloning Model')