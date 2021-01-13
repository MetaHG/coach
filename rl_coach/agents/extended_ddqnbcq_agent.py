from collections import OrderedDict
from typing import Union, List, Tuple

import numpy as np
from copy import deepcopy

from rl_coach.agents.dqn_agent import DQNNetworkParameters
from rl_coach.agents.ddqn_bcq_agent import DDQNBCQAlgorithmParameters, DDQNBCQAgent, DDQNBCQAgentParameters

from rl_coach.core_types import Batch
from rl_coach.off_policy_evaluators.extended_ope_manager import ExtendedOpeManager
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
from rl_coach.core_types import RunPhase

class ExtendedDDQNBCQAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()


class ExtendedDDQNBCQAgentParameters(DDQNBCQAgentParameters):
    def __init__(self):
        super().__init__()
        self.debug = False
        self.ope_manager = ExtendedOpeManager()

    @property
    def path(self):
        return 'rl_coach.agents.extended_ddqnbcq_agent:ExtendedDDQNBCQAgent'


class ExtendedDDQNBCQAgent(DDQNBCQAgent):
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



#TODO: Agent could be running OPEs for all given policies (softmax q, behavior cloning, random) at once
# For this, one would need to extend the extendedOPEManager (gather_static_shared_stats, _prepare_ope_shared_stats and evaluate)
# We would also need to redefine update_log accordingly and run_off_policy_evaluation to include the new stats for each policy
# The agent constructor should include:
#      self.ope_manager = ExtendedOPEManager()
#      self.debug = agent_params.debug
#      self.behavior_cloning = True
#      self.q_softmax = True
#      self.fixed_policies = {'rnd':[0.1, 0.1, ..., 0.1], 'hello':[...], ..., 'world':[...]}
#
# Could avoid training the behavior cloning model if self.behavior_cloning = False, but would require to extend a bit further our extended_batch_rl_graph_manager.py accordingly