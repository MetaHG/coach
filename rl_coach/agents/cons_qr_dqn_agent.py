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

from rl_coach.agents.qr_dqn_agent import QuantileRegressionDQNAlgorithmParameters, QuantileRegressionDQNNetworkParameters, QuantileRegressionDQNAgentParameters
from rl_coach.agents.qr_dqn_agent import QuantileRegressionDQNAgent
from rl_coach.architectures.custom_head_parameters import ConservativeQuantileRegressionQHeadParameters
from rl_coach.core_types import StateType
from rl_coach.schedules import LinearSchedule


class ConservativeQuantileRegressionDQNNetworkParameters(QuantileRegressionDQNNetworkParameters):
    def __init__(self, min_q_weight=0.0):
        super().__init__()
        self.heads_parameters = [ConservativeQuantileRegressionQHeadParameters(min_q_weight=min_q_weight)]


class ConservativeQuantileRegressionDQNAlgorithmParameters(QuantileRegressionDQNAlgorithmParameters):
    def __init__(self):
        super().__init__()


class ConservativeQuantileRegressionDQNAgentParameters(QuantileRegressionDQNAgentParameters):
    def __init__(self, min_q_weight=0.0):
        super().__init__()
        self.algorithm = ConservativeQuantileRegressionDQNAlgorithmParameters()
        self.network_wrappers = {"main": ConservativeQuantileRegressionDQNNetworkParameters(min_q_weight=min_q_weight)}

    @property
    def path(self):
        return 'rl_coach.agents.cons_qr_dqn_agent:ConservativeQuantileRegressionDQNAgent'


# CQL Quantile Regression Deep Q Network - https://github.com/aviralkumar2907/CQL
class ConservativeQuantileRegressionDQNAgent(QuantileRegressionDQNAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

