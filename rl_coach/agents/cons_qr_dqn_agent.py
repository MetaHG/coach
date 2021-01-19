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
from collections import OrderedDict
from typing import Union, List, Tuple

import numpy as np

from rl_coach.agents.value_optimization_agent import ValueOptimizationAgent
from rl_coach.agents.dqn_agent import DQNAgentParameters, DQNNetworkParameters, DQNAlgorithmParameters

from rl_coach.agents.qr_dqn_agent import QuantileRegressionDQNAlgorithmParameters, QuantileRegressionDQNNetworkParameters, QuantileRegressionDQNAgentParameters
from rl_coach.agents.qr_dqn_agent import QuantileRegressionDQNAgent
from rl_coach.architectures.custom_head_parameters import ConservativeQuantileRegressionQHeadParameters
from rl_coach.core_types import StateType
from rl_coach.schedules import LinearSchedule
from rl_coach.core_types import RunPhase
from rl_coach.logger import screen
from rl_coach.filters.filter import NoInputFilter

from rl_coach.off_policy_evaluators.extended_ope_manager import ExtendedOpeManager


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
        self.network_wrappers = {"main": ConservativeQuantileRegressionDQNNetworkParameters()}
        
        # OPE Extension
        self.debug = False
        self.ope_manager = ExtendedOpeManager()
        
        # Conservative Q-Learning
        self.min_q_weight = 0.0
        self.set_q_weight(min_q_weight)

    @property
    def path(self):
        return 'rl_coach.agents.cons_qr_dqn_agent:ConservativeQuantileRegressionDQNAgent'
    
    def set_q_weight(self, min_q_weight=0.0):
        self.min_q_weight = min_q_weight
        self.update_q_weight()
        
    def update_q_weight(self):
        self.network_wrappers['main'].heads_parameters = [ConservativeQuantileRegressionQHeadParameters(min_q_weight=self.min_q_weight)]


# CQL Quantile Regression Deep Q Network - https://github.com/aviralkumar2907/CQL
class ConservativeQuantileRegressionDQNAgent(QuantileRegressionDQNAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        
        self.debug = agent_parameters.debug
        self.ope_manager = agent_parameters.ope_manager
        
        
    def update_log(self):
        """
        Updates the episodic log file with all the signal values from the most recent episode.
        Additional signals for logging can be set by the creating a new signal using self.register_signal,
        and then updating it with some internal agent values.

        :return: None
        """
        # log all the signals to file
        current_time = self.get_current_time()
        self.agent_logger.set_current_time(current_time)
        self.agent_logger.create_signal_value('Training Iter', self.training_iteration)
        self.agent_logger.create_signal_value('Episode #', self.current_episode)
        self.agent_logger.create_signal_value('Epoch', self.training_epoch)
        self.agent_logger.create_signal_value('In Heatup', int(self._phase == RunPhase.HEATUP))
        self.agent_logger.create_signal_value('ER #Transitions', self.call_memory('num_transitions'))
        self.agent_logger.create_signal_value('ER #Episodes', self.call_memory('length'))
        self.agent_logger.create_signal_value('Episode Length', self.current_episode_steps_counter)
        self.agent_logger.create_signal_value('Total steps', self.total_steps_counter)
        self.agent_logger.create_signal_value("Epsilon", np.mean(self.exploration_policy.get_control_param()))
        self.agent_logger.create_signal_value("Shaped Training Reward", self.total_shaped_reward_in_current_episode
                                   if self._phase == RunPhase.TRAIN else np.nan)
        self.agent_logger.create_signal_value("Training Reward", self.total_reward_in_current_episode
                                   if self._phase == RunPhase.TRAIN else np.nan)

        self.agent_logger.create_signal_value('Update Target Network', 0, overwrite=False)
        self.agent_logger.update_wall_clock_time(current_time)

        # The following signals are created with meaningful values only when an evaluation phase is completed.
        # Creating with default NaNs for any HEATUP/TRAIN/TEST episode which is not the last in an evaluation phase
        self.agent_logger.create_signal_value('Evaluation Reward', np.nan, overwrite=False)
        self.agent_logger.create_signal_value('Shaped Evaluation Reward', np.nan, overwrite=False)
        self.agent_logger.create_signal_value('Success Rate', np.nan, overwrite=False)
        self.agent_logger.create_signal_value('Inverse Propensity Score', np.nan, overwrite=False)
        self.agent_logger.create_signal_value('Direct Method Reward', np.nan, overwrite=False)
        self.agent_logger.create_signal_value('Doubly Robust', np.nan, overwrite=False)
        self.agent_logger.create_signal_value('Weighted Importance Sampling', np.nan, overwrite=False)
        self.agent_logger.create_signal_value('Sequential Doubly Robust', np.nan, overwrite=False)

        # Add custom OPE signals
        self.agent_logger.create_signal_value('Infinite Horizon Doubly Robust', np.nan, overwrite=False)
        self.agent_logger.create_signal_value('Weighted Doubly Robust', np.nan, overwrite=False)

        for signal in self.episode_signals:
            self.agent_logger.create_signal_value("{}/Mean".format(signal.name), signal.get_mean())
            self.agent_logger.create_signal_value("{}/Stdev".format(signal.name), signal.get_stdev())
            self.agent_logger.create_signal_value("{}/Max".format(signal.name), signal.get_max())
            self.agent_logger.create_signal_value("{}/Min".format(signal.name), signal.get_min())

        # Add debug signals
        if self.debug:
            self.agent_logger.create_signal_value('wi_sum_across_episodes', np.nan, overwrite=False)
            self.agent_logger.create_signal_value('wi_per_episode', [], overwrite=False)
            self.agent_logger.create_signal_value('wi_per_transition', [], overwrite=False)
            self.agent_logger.create_signal_value('seq_dr_per_episode', [], overwrite=False)
            self.agent_logger.create_signal_value('seq_dr_per_transition', [], overwrite=False)
            self.agent_logger.create_signal_value('rho_per_transition', [], overwrite=False)
            self.agent_logger.create_signal_value('reward_per_transition', [], overwrite=False)
            self.agent_logger.create_signal_value('v_value_per_transition', [], overwrite=False)
            self.agent_logger.create_signal_value('q_value_per_transition', [], overwrite=False)

        # dump
        if self.current_episode % self.ap.visualization.dump_signals_to_csv_every_x_episodes == 0:
            self.agent_logger.dump_output_csv()

    def run_off_policy_evaluation(self):
        """
        Run the off-policy evaluation estimators to get a prediction for the performance of the current policy based on
        an evaluation dataset, which was collected by another policy(ies).
        :return: None
        """
        assert self.ope_manager

        if not isinstance(self.pre_network_filter, NoInputFilter) and len(self.pre_network_filter.reward_filters) != 0:
            raise ValueError("Defining a pre-network reward filter when OPEs are calculated will result in a mismatch "
                             "between q values (which are scaled), and actual rewards, which are not. It is advisable "
                             "to use an input_filter, if possible, instead, which will filter the transitions directly "
                             "in the replay buffer, affecting both the q_values and the rewards themselves. ")

        if self.debug:
            ips, dm, dr, seq_dr, wis, inf_dr, wdr, \
            wis, wi_sum_across_episodes, wi_per_episode, wi_per_transition, \
            seq_dr, seq_dr_per_episode, seq_dr_per_transition, rho_per_transition, \
            reward_per_transition, v_value_per_transition, q_value_per_transition \
                = self.ope_manager.evaluate(
                    evaluation_dataset_as_episodes=self.memory.evaluation_dataset_as_episodes,
                    evaluation_dataset_as_transitions=self.memory.evaluation_dataset_as_transitions,
                    batch_size=self.ap.network_wrappers['main'].batch_size,
                    discount_factor=self.ap.algorithm.discount,
                    q_network=self.networks['main'].online_network,
                    network_keys=list(self.ap.network_wrappers['main'].input_embedders_parameters.keys()))

        else:
            ips, dm, dr, seq_dr, wis, inf_dr, wdr \
                = self.ope_manager.evaluate(
                    evaluation_dataset_as_episodes=self.memory.evaluation_dataset_as_episodes,
                    evaluation_dataset_as_transitions=self.memory.evaluation_dataset_as_transitions,
                    batch_size=self.ap.network_wrappers['main'].batch_size,
                    discount_factor=self.ap.algorithm.discount,
                    q_network=self.networks['main'].online_network,
                    network_keys=list(self.ap.network_wrappers['main'].input_embedders_parameters.keys()))

        # get the estimators out to the screen
        log = OrderedDict()
        log['Epoch'] = self.training_epoch
        log['IPS'] = ips
        log['DM'] = dm
        log['DR'] = dr
        log['WIS'] = wis
        log['Sequential-DR'] = seq_dr
        log['Infinite-DR'] = inf_dr
        log['WDR'] = wdr
        screen.log_dict(log, prefix='Off-Policy Evaluation')

        # get the estimators out to dashboard
        self.agent_logger.set_current_time(self.get_current_time() + 1)
        self.agent_logger.create_signal_value('Inverse Propensity Score', ips)
        self.agent_logger.create_signal_value('Direct Method Reward', dm)
        self.agent_logger.create_signal_value('Doubly Robust', dr)
        self.agent_logger.create_signal_value('Sequential Doubly Robust', seq_dr)
        self.agent_logger.create_signal_value('Weighted Importance Sampling', wis)
        self.agent_logger.create_signal_value('Sequential Doubly Robust', seq_dr)
        self.agent_logger.create_signal_value('Infinite Horizon Doubly Robust', inf_dr)
        self.agent_logger.create_signal_value('Weighted Doubly Robust', wdr)
        
        if self.debug:
            self.agent_logger.create_signal_value('wi_sum_across_episodes', wi_sum_across_episodes)
            self.agent_logger.create_signal_value('wi_per_episode', wi_per_episode)
            self.agent_logger.create_signal_value('wi_per_transition', wi_per_transition)

            self.agent_logger.create_signal_value('seq_dr_per_episode', seq_dr_per_episode)
            self.agent_logger.create_signal_value('seq_dr_per_transition', seq_dr_per_transition)
            self.agent_logger.create_signal_value('rho_per_transition', rho_per_transition)
            self.agent_logger.create_signal_value('reward_per_transition', reward_per_transition)
            self.agent_logger.create_signal_value('v_value_per_transition', v_value_per_transition)
            self.agent_logger.create_signal_value('q_value_per_transition', q_value_per_transition)


