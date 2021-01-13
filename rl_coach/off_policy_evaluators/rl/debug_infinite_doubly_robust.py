from typing import List
import numpy as np

from rl_coach.core_types import Episode


class DebugInfiniteDoublyRobust(object):

    @staticmethod
    def evaluate(evaluation_dataset_as_episodes: List[Episode], discount_factor: float) -> float:
        """
        Run the off-policy evaluator to get a score for the goodness of the new policy, based on the dataset,
        which was collected using other policy(ies).
        Paper: https://cs.stanford.edu/people/ebrun/pdfs/thomas2016data.pdf

        :return: the evaluation score
        """

        # Infinite Doubly Robust
        per_episode_seq_dr = []
        
        rho_per_episode = []
        reward_per_episode = []
        q_value_per_episode = []
        v_value_per_episode = []
        dr_value_per_episode = []
        behavior_prob_per_episode = []
        target_prob_per_episode = []

        for episode in evaluation_dataset_as_episodes:
            episode_seq_dr = 0
            previous_rho = 1
            discount_factor_at_step = 1
            
            rho_per_transition = []
            reward_per_transition = []
            q_value_per_transition = []
            v_value_per_transition = []
            dr_value_per_transition = []
            behavior_prob_per_transition = []
            target_prob_per_transition = []
            
            for transition in episode.transitions:
                rho = previous_rho * transition.info['softmax_policy_prob'][transition.action] / \
                                     transition.info['all_action_probabilities'][transition.action]

                immediate_reward = rho * transition.reward
                expected_return_error = (rho * transition.info['q_value'][transition.action] 
                                            - previous_rho * transition.info['v_value_q_model_based'])

                episode_seq_dr += discount_factor_at_step * (immediate_reward - expected_return_error)
                
                # Update
                discount_factor_at_step *= discount_factor
                previous_rho = rho
                
                # Debug outputs
                rho_per_transition.append(rho)
                #reward_per_transition.append(transition.reward)
                #q_value_per_transition.append(transition.info['q_value'][transition.action])
                #v_value_per_transition.append(transition.info['v_value_q_model_based'])
                dr_value_per_transition.append(episode_seq_dr)
                behavior_prob_per_transition.append(transition.info['all_action_probabilities'][transition.action])
                target_prob_per_transition.append(transition.info['softmax_policy_prob'][transition.action])

            
            per_episode_seq_dr.append(episode_seq_dr)
            
            # Debug outputs
            rho_per_episode.append(rho_per_transition)
            #reward_per_episode.append(reward_per_transition)
            #q_value_per_episode.append(q_value_per_transition)
            #v_value_per_episode.append(v_value_per_transition)
            dr_value_per_episode.append(dr_value_per_transition)

            behavior_prob_per_episode.append(behavior_prob_per_transition)
            target_prob_per_episode.append(target_prob_per_transition)

        seq_dr = np.array(per_episode_seq_dr).mean()
        
        # reward, q_value and v_value are provided by the sequential_dr debug
        # return seq_dr, rho_per_episode, reward_per_episode, q_value_per_episode, v_value_per_episode, dr_value_per_episode
        return seq_dr, per_episode_seq_dr, rho_per_episode, dr_value_per_episode, behavior_prob_per_episode, target_prob_per_episode