#
# Copyright (c) 2017 Intel Corporation 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from copy import copy
from typing import Union

import numpy as np

from rl_coach.agents.value_optimization_agent import ValueOptimizationAgent
from rl_coach.agents.dqn_agent import DQNAgentParameters, DQNNetworkParameters, DQNAlgorithmParameters

from rl_coach.agents.ddqn_agent import DDQNAgentParameters, DDQNAgent
from rl_coach.agents.dqn_agent import DQNNetworkParameters, DQNAlgorithmParameters

from rl_coach.architectures.custom_head_parameters import ConservativeQuantileRegressionQHeadParameters
from rl_coach.core_types import StateType
from rl_coach.schedules import LinearSchedule


class ConservativeDDQNNetworkParameters(DQNNetworkParameters):
    def __init__(self, min_q_weight=0.0):
        super().__init__()
        self.heads_parameters = [ConservativeQuantileRegressionQHeadParameters(min_q_weight=min_q_weight)] #TODO


class ConservativeDDQNAlgorithmParameters(DQNAlgorithmParameters):
    def __init__(self):
        super().__init__()


class ConservativeDDQNAgentParameters(DDQNAgentParameters):
    def __init__(self, min_q_weight=0.0):
        super().__init__()
        self.algorithm = ConservativeDDQNAlgorithmParameters()
        self.network_wrappers = {"main": ConservativeDDQNNetworkParameters(min_q_weight=min_q_weight)}

    @property
    def path(self):
        return 'rl_coach.agents.cons_ddqn_agent:ConservativeDDQNAgent'


# CQL Quantile Regression Deep Q Network - https://github.com/aviralkumar2907/CQL
class ConservativeDDQNAgent(DDQNAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

    def learn_from_batch(self, batch):
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        # for the action we actually took, the error is:
        # TD error = r + discount*max(q_st_plus_1) - q_st
        # # for all other actions, the error is 0
        q_st_plus_1, TD_targets = self.networks['main'].parallel_prediction([
            (self.networks['main'].target_network, batch.next_states(network_keys)),
            (self.networks['main'].online_network, batch.states(network_keys))
        ])

        selected_actions = self.select_actions(batch.next_states(network_keys), q_st_plus_1)

        # add Q value samples for logging
        self.q_values.add_sample(TD_targets)

        #  only update the action that we have actually done in this transition
        TD_errors = []
        for i in range(batch.size):
            new_target = batch.rewards()[i] +\
                         (1.0 - batch.game_overs()[i]) * self.ap.algorithm.discount * q_st_plus_1[i][selected_actions[i]]
            TD_errors.append(np.abs(new_target - TD_targets[i, batch.actions()[i]]))
            TD_targets[i, batch.actions()[i]] = new_target

        # update errors in prioritized replay buffer
        importance_weights = self.update_transition_priorities_and_get_weights(TD_errors, batch)

        result = self.networks['main'].train_and_sync_networks({
            **batch.states(network_keys), 
            'output_0_0':batch.actions()
            }, TD_targets,
            importance_weights=importance_weights)

        total_loss, losses, unclipped_grads = result[:3]

        return total_loss, losses, unclipped_grads

