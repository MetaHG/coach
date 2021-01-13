#
# Copyright (c) 2019 Intel Corporation
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
import math
from collections import namedtuple

import numpy as np
from typing import List

from rl_coach.off_policy_evaluators.ope_manager import OpeManager
from rl_coach.architectures.architecture import Architecture
from rl_coach.core_types import Episode, Batch
from rl_coach.off_policy_evaluators.bandits.doubly_robust import DoublyRobust
from rl_coach.off_policy_evaluators.rl.sequential_doubly_robust import SequentialDoublyRobust
from rl_coach.off_policy_evaluators.rl.debug_sequential_doubly_robust import DebugSequentialDoublyRobust

from rl_coach.off_policy_evaluators.rl.infinite_doubly_robust import InfiniteDoublyRobust
from rl_coach.off_policy_evaluators.rl.debug_infinite_doubly_robust import DebugInfiniteDoublyRobust
from rl_coach.off_policy_evaluators.rl.weighted_doubly_robust import WeightedDoublyRobust
from rl_coach.off_policy_evaluators.rl.debug_weighted_doubly_robust import DebugWeightedDoublyRobust

from rl_coach.core_types import Transition

from rl_coach.off_policy_evaluators.rl.debug_weighted_importance_sampling import DebugWeightedImportanceSampling
from rl_coach.off_policy_evaluators.rl.weighted_importance_sampling import WeightedImportanceSampling

OpeSharedStats = namedtuple("OpeSharedStats", ['all_reward_model_rewards', 'all_policy_probs',
                                               'all_v_values_reward_model_based', 'all_rewards', 'all_actions',
                                               'all_old_policy_probs', 'new_policy_prob', 'rho_all_dataset'])
OpeEstimation = namedtuple("OpeEstimation", ['ips', 'dm', 'dr', 'seq_dr', 'wis', 'inf_dr', 'wdr'])
DebugOpeEstimation = namedtuple("DebugOpeEstimaton", ['ips', 'dm', 'dr', 'seq_dr', 'wis', 'inf_dr', 'wdr',
                                                      'wi_sum_across_episodes', 'wi_per_episode', 'wi_per_transition',
                                                      'seq_dr_per_episode', 'seq_dr_per_transition', 'rho_per_transition', 
                                                      'reward_per_transition', 'v_value_per_transition', 'q_value_per_transition', 
                                                      'behavior_prob_per_transition', 'target_prob_per_transition',
                                                      'inf_dr_per_episode', 'inf_rho_per_transition', 'inf_dr_per_transition',
                                                      'wdr_per_episode', 'wdr_wi_per_transition', 'wdr_sum_rho_per_transition', 'wdr_per_transition'])

class ExtendedOpeManager(OpeManager):
    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug
        self.random_policy = False
        self.debug_wdr = False

        if self.debug:
            self.debug_weighted_importance_sampling = DebugWeightedImportanceSampling()
            self.debug_sequential_doubly_robust = DebugSequentialDoublyRobust()
            self.debug_infinite_doubly_robust = DebugInfiniteDoublyRobust()
            self.debug_weighted_doubly_robust = DebugWeightedDoublyRobust()

        self.infinite_horizon_doubly_robust = InfiniteDoublyRobust()
        self.weighted_doubly_robust = WeightedDoublyRobust()
        
    
    def set_debug(self):
        self.debug = True
        self.debug_weighted_importance_sampling = DebugWeightedImportanceSampling()
        self.debug_sequential_doubly_robust = DebugSequentialDoublyRobust()
        self.debug_infinite_doubly_robust = DebugInfiniteDoublyRobust()
        self.debug_weighted_doubly_robust = DebugWeightedDoublyRobust()

    
    def set_random_policy(self):
        self.random_policy = True


    def set_debug_wdr(self):
        self.debug_wdr = True


    def _prepare_ope_shared_stats(self, evaluation_dataset_as_transitions: List[Transition], batch_size: int,
                                  q_network: Architecture, network_keys: List) -> OpeSharedStats:
        """
        Do the preparations needed for different estimators.
        Some of the calcuations are shared, so we centralize all the work here.

        :param evaluation_dataset_as_transitions: The evaluation dataset in the form of transitions.
        :param batch_size: The batch size to use.
        :param reward_model: A reward model to be used by DR
        :param q_network: The Q network whose its policy we evaluate.
        :param network_keys: The network keys used for feeding the neural networks.
        :return:
        """

        assert self.is_gathered_static_shared_data, "gather_static_shared_stats() should be called once before " \
                                                    "calling _prepare_ope_shared_stats()"
        # IPS
        all_policy_probs = []
        all_v_values_reward_model_based, all_v_values_q_model_based = [], []

        for i in range(math.ceil(len(evaluation_dataset_as_transitions) / batch_size)):
            batch = evaluation_dataset_as_transitions[i * batch_size: (i + 1) * batch_size]
            batch_for_inference = Batch(batch)

            # we always use the first Q head to calculate OPEs. might want to change this in the future.
            # for instance, this means that for bootstrapped dqn we always use the first QHead to calculate the OPEs.
            q_values, sm_values = q_network.predict(batch_for_inference.states(network_keys),
                                                    outputs=[q_network.output_heads[0].q_values,
                                                             q_network.output_heads[0].softmax])

            if self.random_policy:
                sm_values = np.full(sm_values.shape, 1/sm_values.shape[1])

            all_policy_probs.append(sm_values)
            all_v_values_reward_model_based.append(np.sum(all_policy_probs[-1] * self.all_reward_model_rewards[i],
                                                          axis=1))
            all_v_values_q_model_based.append(np.sum(all_policy_probs[-1] * q_values, axis=1))

            for j, t in enumerate(batch):
                t.update_info({
                    'q_value': q_values[j],
                    'softmax_policy_prob': all_policy_probs[-1][j],
                    'v_value_q_model_based': all_v_values_q_model_based[-1][j],
                })

                if self.debug_wdr:
                    t.update_info({
                        'all_action_probabilities': all_policy_probs[-1][j],
                    })

        all_policy_probs = np.concatenate(all_policy_probs, axis=0)
        all_v_values_reward_model_based = np.concatenate(all_v_values_reward_model_based, axis=0)

        # generate model probabilities
        new_policy_prob = all_policy_probs[np.arange(self.all_actions.shape[0]), self.all_actions]
        rho_all_dataset = new_policy_prob / self.all_old_policy_probs

        return OpeSharedStats(self.all_reward_model_rewards, all_policy_probs, all_v_values_reward_model_based,
                              self.all_rewards, self.all_actions, self.all_old_policy_probs, new_policy_prob,
                              rho_all_dataset)


    def evaluate(self, evaluation_dataset_as_episodes: List[Episode], evaluation_dataset_as_transitions: List[Transition], batch_size: int,
                 discount_factor: float, q_network: Architecture, network_keys: List) -> OpeEstimation:
        """
        Run all the OPEs and get estimations of the current policy performance based on the evaluation dataset.

        :param evaluation_dataset_as_episodes: The evaluation dataset in a form of episodes.
        :param evaluation_dataset_as_transitions: The evaluation dataset in a form of transitions.
        :param batch_size: Batch size to use for the estimators.
        :param discount_factor: The standard RL discount factor.
        :param reward_model: A reward model to be used by DR
        :param q_network: The Q network whose its policy we evaluate.
        :param network_keys: The network keys used for feeding the neural networks.

        :return: An OpeEstimation tuple which groups together all the OPE estimations
        """
        ope_shared_stats = self._prepare_ope_shared_stats(evaluation_dataset_as_transitions, batch_size, q_network,
                                                          network_keys)

        ips, dm, dr = self.doubly_robust.evaluate(ope_shared_stats)

        inf_dr = self.infinite_horizon_doubly_robust.evaluate(evaluation_dataset_as_episodes, discount_factor)
        wdr = self.weighted_doubly_robust.evaluate(evaluation_dataset_as_episodes, discount_factor)

        if self.debug:
            # Careful with the probabilities from sequential_doubly_robust as they are in the reversed order of transitions.
            seq_dr, seq_dr_per_episode, seq_dr_per_transition, rho_per_transition, reward_per_transition, v_value_per_transition, q_value_per_transition, behavior_prob_per_transition, target_prob_per_transition \
                = self.debug_sequential_doubly_robust.evaluate(evaluation_dataset_as_episodes, discount_factor)
            wis, wi_sum_across_episodes, wi_per_episode, wi_per_transition = self.debug_weighted_importance_sampling.evaluate(evaluation_dataset_as_episodes)
            inf_dr, inf_dr_per_episode, inf_rho_per_transition, inf_dr_per_transition, _, _ = self.debug_infinite_doubly_robust.evaluate(evaluation_dataset_as_episodes, discount_factor)
            wdr, wdr_per_episode, wdr_wi_per_transition, wdr_sum_rho_per_transition, wdr_per_transition = self.debug_weighted_doubly_robust.evaluate(evaluation_dataset_as_episodes, discount_factor)
            return DebugOpeEstimation(ips, dm, dr, seq_dr, wis, inf_dr, wdr, 
                                      wi_sum_across_episodes, wi_per_episode, wi_per_transition, # WIS Debug
                                      seq_dr_per_episode, seq_dr_per_transition, rho_per_transition, # SeqDR Debug
                                      reward_per_transition, v_value_per_transition, q_value_per_transition, # General/SeqDR Debug
                                      behavior_prob_per_transition, target_prob_per_transition, # General/SeqDR Debug
                                      inf_dr_per_episode, inf_rho_per_transition, inf_dr_per_transition, # InfDR Debug
                                      wdr_per_episode, wdr_wi_per_transition, wdr_sum_rho_per_transition, wdr_per_transition) # WDR Debug
        else:
            wis = self.weighted_importance_sampling.evaluate(evaluation_dataset_as_episodes)
            seq_dr = self.sequential_doubly_robust.evaluate(evaluation_dataset_as_episodes, discount_factor)
            return OpeEstimation(ips, dm, dr, seq_dr, wis, inf_dr, wdr)

