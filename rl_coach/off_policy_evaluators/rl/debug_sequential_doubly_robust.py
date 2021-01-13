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
from typing import List
import numpy as np

from rl_coach.core_types import Episode


class DebugSequentialDoublyRobust(object):

    @staticmethod
    def evaluate(evaluation_dataset_as_episodes: List[Episode], discount_factor: float):
        """
        Run the off-policy evaluator to get a score for the goodness of the new policy, based on the dataset,
        which was collected using other policy(ies).
        When the epsiodes are of changing lengths, this estimator might prove problematic due to its nature of recursion
        of adding rewards up to the end of the episode (horizon). It will probably work best with episodes of fixed
        length.
        Paper: https://arxiv.org/pdf/1511.03722.pdf

        :return: the evaluation score
        """

        # Sequential Doubly Robust
        
        per_episode_seq_dr = []
        
        # Debug
        per_transition_seq_dr = []
        per_transition_rho = []
        per_transition_reward = []
        per_transition_v_value = []
        per_transition_q_value = []
        per_transition_behavior_prob = []
        per_transition_target_prob = []

        for episode in evaluation_dataset_as_episodes:
            episode_seq_dr = 0

            episode_rho = []
            episode_reward = []
            episode_seq_dr_list = []
            episode_v_value = []
            episode_q_value = []
            episode_behavior_prob = []
            episode_target_prob = []

            for transition in reversed(episode.transitions):
                rho = transition.info['softmax_policy_prob'][transition.action] / \
                      transition.info['all_action_probabilities'][transition.action]
                episode_seq_dr = transition.info['v_value_q_model_based'] + rho * (transition.reward + discount_factor
                                                                                   * episode_seq_dr -
                                                                                   transition.info['q_value'][
                                                                                       transition.action])
                # Debug
                episode_seq_dr_list.append(episode_seq_dr)
                episode_rho.append(rho)
                episode_reward.append(transition.reward)
                episode_v_value.append(transition.info['v_value_q_model_based'])
                episode_q_value.append(transition.info['q_value'][transition.action])
                episode_behavior_prob.append(transition.info['all_action_probabilities'][transition.action])
                episode_target_prob.append(transition.info['softmax_policy_prob'][transition.action])

            per_episode_seq_dr.append(episode_seq_dr)
            
            # Debug
            per_transition_rho.append(episode_rho)
            per_transition_reward.append(episode_reward)
            per_transition_seq_dr.append(episode_seq_dr_list)
            per_transition_v_value.append(episode_v_value)
            per_transition_q_value.append(episode_q_value)
            per_transition_behavior_prob.append(episode_behavior_prob)
            per_transition_target_prob.append(episode_target_prob)
            
            
            

        seq_dr = np.array(per_episode_seq_dr).mean()

        return seq_dr, per_episode_seq_dr, per_transition_seq_dr, per_transition_rho, per_transition_reward, per_transition_v_value, per_transition_q_value, per_transition_behavior_prob, per_transition_target_prob
