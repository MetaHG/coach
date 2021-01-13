from typing import List
import numpy as np

from rl_coach.core_types import Episode


class InfiniteDoublyRobust(object):

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

        for episode in evaluation_dataset_as_episodes:
            episode_seq_dr = 0
            previous_rho = 1
            discount_factor_at_step = 1

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
            
            
            per_episode_seq_dr.append(episode_seq_dr)

        seq_dr = np.array(per_episode_seq_dr).mean()

        return seq_dr