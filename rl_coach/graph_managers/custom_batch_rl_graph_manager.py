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
from copy import deepcopy
from typing import Tuple, List, Union

from rl_coach.architectures.head_parameters import PolicyHeadParameters
from rl_coach.agents.ddqn_bcq_agent import DDQNBCQAgentParameters
from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.agents.nec_agent import NECAgentParameters
from rl_coach.architectures.network_wrapper import NetworkWrapper
from rl_coach.base_parameters import AgentParameters, VisualizationParameters, TaskParameters, \
    PresetValidationParameters
from rl_coach.core_types import RunPhase, TotalStepsCounter, TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.environment import EnvironmentParameters, Environment
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.batch_rl_graph_manager import BatchRLGraphManager

from rl_coach.level_manager import LevelManager
from rl_coach.logger import screen
from rl_coach.schedules import LinearSchedule
from rl_coach.spaces import SpacesDefinition
from rl_coach.utils import short_dynamic_import

from rl_coach.memories.episodic import EpisodicExperienceReplayParameters

from rl_coach.core_types import TimeTypes


class CustomBatchRLGraphManager(BatchRLGraphManager):
    """
    A batch RL graph manager creates a scenario of learning from a dataset without a simulator.

    If an environment is given (useful either for research purposes, or for experimenting with a toy problem before
    actually working with a real dataset), we can use it in order to collect a dataset to later be used to train the
    actual agent. The collected dataset, in this case, can be collected either by randomly acting in the environment
    (only running in heatup), or alternatively by training a different agent in the environment and using its collected
    data as a dataset. If an experience generating agent parameters are given, we will instantiate this agent and use it
     in order to train on the environment and then use this dataset to actually train an agent. Otherwise, we will
     collect a random dataset.
    :param agent_params: the parameters of the agent to train using batch RL
    :param env_params: [optional] environment parameters, for cases where we want to first collect a dataset
    :param vis_params: visualization parameters
    :param preset_validation_params: preset validation parameters, to be used for testing purposes
    :param name: graph name
    :param spaces_definition: when working with a dataset, we need to get a description of the actual state and action
                              spaces of the problem
    :param reward_model_num_epochs: the number of epochs to go over the dataset for training a reward model for the
                                    'direct method' and 'doubly robust' OPE methods.
    :param train_to_eval_ratio: percentage of the data transitions to be used for training vs. evaluation. i.e. a value
                                of 0.8 means ~80% of the transitions will be used for training and ~20% will be used for
                                 evaluation using OPE.
    :param experience_generating_agent_params: [optional] parameters of an agent to be trained vs. an environment, whose
                                               his collected experience will be used to train the acutal (another) agent
    :param experience_generating_schedule_params: [optional] graph scheduling parameters for training the experience
                                                  generating agent
    """
    def __init__(self, agent_params: AgentParameters,
                 env_params: Union[EnvironmentParameters, None],
                 schedule_params: ScheduleParameters,
                 vis_params: VisualizationParameters = VisualizationParameters(),
                 preset_validation_params: PresetValidationParameters = PresetValidationParameters(),
                 name='batch_rl_graph', spaces_definition: SpacesDefinition = None, reward_model_num_epochs: int = 100, behavior_cloning_model_num_epochs: int = 100,
                 train_to_eval_ratio: float = 0.8, experience_generating_agent_params: AgentParameters = None,
                 experience_generating_schedule_params: ScheduleParameters = None):

        super().__init__(agent_params, env_params, schedule_params, vis_params, preset_validation_params, name, spaces_definition, reward_model_num_epochs, 
                         train_to_eval_ratio, experience_generating_agent_params, experience_generating_schedule_params)
        
        self.behavior_cloning_model_num_epochs = behavior_cloning_model_num_epochs


    def _create_graph(self, task_parameters: TaskParameters) -> Tuple[List[LevelManager], List[Environment]]:
        assert self.agent_params.memory.load_memory_from_file_path or self.env_params, \
            "BatchRL requires either a dataset to train from or an environment to collect a dataset from. "
        if self.env_params:
            # environment loading
            self.env_params.seed = task_parameters.seed
            self.env_params.experiment_path = task_parameters.experiment_path
            env = short_dynamic_import(self.env_params.path)(**self.env_params.__dict__,
                                                             visualization_parameters=self.visualization_parameters)
        else:
            env = None

        # Only DQN variants and NEC are supported at this point.
        assert(isinstance(self.agent_params, DQNAgentParameters) or isinstance(self.agent_params, NECAgentParameters))
        # Only Episodic memories are supported,
        # for evaluating the sequential doubly robust estimator
        assert(isinstance(self.agent_params.memory, EpisodicExperienceReplayParameters))

        # agent loading
        self.agent_params.task_parameters = task_parameters  # TODO: this should probably be passed in a different way
        self.agent_params.name = "agent"
        self.agent_params.is_batch_rl_training = True
        self.agent_params.network_wrappers['main'].should_get_softmax_probabilities = True

        if 'reward_model' not in self.agent_params.network_wrappers:
            # user hasn't defined params for the reward model. we will use the same params as used for the 'main'
            # network.
            self.agent_params.network_wrappers['reward_model'] = deepcopy(self.agent_params.network_wrappers['main'])

        if 'behavior_cloning_model' not in self.agent_params.network_wrappers:
            # user hasn't defined params for the behavior cloning model. we will use the same params as used for the 'main'
            # network.
            self.agent_params.network_wrappers['behavior_cloning_model'] = deepcopy(self.agent_params.network_wrappers['main'])
            self.agent_params.network_wrappers['behavior_cloning_model'].heads_parameters = [PolicyHeadParameters()]
            self.agent_params.network_wrappers['behavior_cloning_model'].optimizer_type = 'Adam'
            self.agent_params.network_wrappers['behavior_cloning_model'].batch_size = 32
            self.agent_params.network_wrappers['behavior_cloning_model'].replace_mse_with_huber_loss = False
            self.agent_params.network_wrappers['behavior_cloning_model'].create_target_network = False

        self.agent = short_dynamic_import(self.agent_params.path)(self.agent_params)
        agents = {'agent': self.agent}

        if not self.is_collecting_random_dataset:
            self.experience_generating_agent_params.visualization.dump_csv = False
            self.experience_generating_agent_params.task_parameters = task_parameters
            self.experience_generating_agent_params.name = "experience_gen_agent"
            self.experience_generating_agent_params.network_wrappers['main'].should_get_softmax_probabilities = True

            # we need to set these manually as these are usually being set for us only for the default agent
            self.experience_generating_agent_params.input_filter = self.agent_params.input_filter
            self.experience_generating_agent_params.output_filter = self.agent_params.output_filter

            self.experience_generating_agent = short_dynamic_import(
                self.experience_generating_agent_params.path)(self.experience_generating_agent_params)

            agents['experience_generating_agent'] = self.experience_generating_agent

        if not env and not self.agent_params.memory.load_memory_from_file_path:
            screen.warning("A BatchRLGraph requires setting a dataset to load into the agent's memory or alternatively "
                           "using an environment to create a (random) dataset from. This agent should only be used for "
                           "inference. ")
        # set level manager
        # - although we will be using each agent separately, we have to have both agents initialized together with the
        #   LevelManager, so to have them both properly initialized
        level_manager = LevelManager(agents=agents,
                                     environment=env, name="main_level",
                                     spaces_definition=self.spaces_definition)
        if env:
            return [level_manager], [env]
        else:
            return [level_manager], []


    def initialize_ope_models_and_stats(self):
        """
        Improve a reward model of the MDP, to be used for some of the off-policy evaluation (OPE) methods.
        e.g. 'direct method' and 'doubly robust'.
        """
        agent = self.level_managers[0].agents['agent']

        # prepare dataset to be consumed in the expected formats for OPE
        agent.memory.prepare_evaluation_dataset()

        screen.log_title("Training a regression model for estimating MDP behavior (behavior cloning)")
        agent.improve_behavior_cloning_model(epochs=self.behavior_cloning_model_num_epochs)

        screen.log_title("Training a regression model for estimating MDP rewards")
        agent.improve_reward_model(epochs=self.reward_model_num_epochs)

        screen.log_title("Collecting static statistics for OPE")
        agent.ope_manager.gather_static_shared_stats(evaluation_dataset_as_transitions=
                                                     agent.memory.evaluation_dataset_as_transitions,
                                                     batch_size=agent.ap.network_wrappers['main'].batch_size,
                                                     reward_model=agent.networks['reward_model'].online_network,
                                                     behavior_cloning_model=agent.networks['behavior_cloning_model'].online_network,
                                                     network_keys=list(agent.ap.network_wrappers['main'].
                                                                       input_embedders_parameters.keys()))


    def run_off_policy_evaluation(self):
        """
        Run off-policy evaluation estimators to evaluate the trained policy performance against the dataset
        :return:
        """
        self.level_managers[0].agents['agent'].run_off_policy_evaluation()
