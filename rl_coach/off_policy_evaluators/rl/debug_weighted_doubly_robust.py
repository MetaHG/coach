from typing import List
import numpy as np

from rl_coach.core_types import Episode


class DebugWeightedDoublyRobust(object):

    @staticmethod
    def evaluate(evaluation_dataset_as_episodes: List[Episode], discount_factor: float) -> float:
        """
        Run the off-policy evaluator to get a score for the goodness of the new policy, based on the dataset,
        which was collected using other policy(ies).
        Paper: https://cs.stanford.edu/people/ebrun/pdfs/thomas2016data.pdf

        :return: the evaluation score
        """

        # Weighted Doubly Robust


        # Compute the modified weights first
        all_transition_weights = []
        
        for episode in evaluation_dataset_as_episodes:
            # Extend the size of the transition weights array if too small
            current_num_transitions = len(all_transition_weights)
            num_transitions = len(episode.transitions)
            new_size = max(num_transitions, current_num_transitions)
            all_transition_weights += [0] * (new_size - current_num_transitions)

            # Compute the cumulative weights
            rho = 1
            for i, transition in enumerate(episode.transitions):
                rho *= transition.info['softmax_policy_prob'][transition.action] / \
                       transition.info['all_action_probabilities'][transition.action]
                all_transition_weights[i] += rho


        # Compute the WDR value
        per_episode_seq_dr = []
        
        # Debug
        wi_per_episode = []
        sum_rho_per_episode = []
        dr_per_episode = []

        for episode in evaluation_dataset_as_episodes:
            episode_seq_dr = 0
            rho = 1
            previous_wi = 1
            discount_factor_at_step = 1
            
            wi_per_transition = []
            sum_rho_per_transition = []
            dr_per_transition = []

            for i, transition in enumerate(episode.transitions):
                rho *= transition.info['softmax_policy_prob'][transition.action] / \
                       transition.info['all_action_probabilities'][transition.action]

                wi = 0
                if all_transition_weights[i] != 0:
                    wi = rho / all_transition_weights[i]

                immediate_reward = wi * transition.reward
                expected_return_error = (wi * transition.info['q_value'][transition.action] 
                                            - previous_wi * transition.info['v_value_q_model_based'])

                episode_seq_dr += discount_factor_at_step * (immediate_reward - expected_return_error)
                
                # Update 
                discount_factor_at_step *= discount_factor
                previous_wi = wi
                
                # Debug
                wi_per_transition.append(wi)
                sum_rho_per_transition.append(all_transition_weights[i])
                dr_per_transition.append(episode_seq_dr)

            
            per_episode_seq_dr.append(episode_seq_dr)
            
            # Debug
            wi_per_episode.append(wi_per_transition)
            sum_rho_per_episode.append(sum_rho_per_transition)
            dr_per_episode.append(dr_per_transition)

        seq_dr = sum(per_episode_seq_dr)

        return seq_dr, per_episode_seq_dr, wi_per_episode, sum_rho_per_episode, dr_per_episode