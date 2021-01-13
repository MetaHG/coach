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

import tensorflow as tf

from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.heads.head import Head
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import QActionStateValue
from rl_coach.spaces import SpacesDefinition, BoxActionSpace, DiscreteActionSpace


class ConservativeQHead(Head):
    def __init__(self, agent_parameters: AgentParameters, spaces: SpacesDefinition, network_name: str,
                 head_idx: int = 0, loss_weight: float = 1., is_local: bool = True, activation_function: str='relu',
                 dense_layer=Dense, output_bias_initializer=None, min_q_weight=0.0):
        super().__init__(agent_parameters, spaces, network_name, head_idx, loss_weight, is_local, activation_function,
                         dense_layer=dense_layer)
        self.name = 'q_values_head'
        if isinstance(self.spaces.action, BoxActionSpace):
            self.num_actions = 1
        elif isinstance(self.spaces.action, DiscreteActionSpace):
            self.num_actions = len(self.spaces.action.actions)
        else:
            raise ValueError(
                'QHead does not support action spaces of type: {class_name}'.format(
                    class_name=self.spaces.action.__class__.__name__,
                )
            )
        self.return_type = QActionStateValue
        if agent_parameters.network_wrappers[self.network_name].replace_mse_with_huber_loss:
            self.loss_type = tf.losses.huber_loss
        else:
            self.loss_type = tf.losses.mean_squared_error

        self.output_bias_initializer = output_bias_initializer

        self.min_q_weight = min_q_weight

    def _build_module(self, input_layer):
        # Standard Q Network
        self.q_values = self.output = self.dense_layer(self.num_actions)\
            (input_layer, name='output', bias_initializer=self.output_bias_initializer)

        # used in batch-rl to estimate a probablity distribution over actions
        self.softmax = self.add_softmax_with_temperature()

        self.actions = tf.placeholder(tf.int32, [None, 1], name="actions")
        self.input = [self.actions]

        ### Add the CQL Loss
        replay_action_one_hot = tf.one_hot(self.actions, self.num_actions, tf.cast(1.0, dtype=tf.float32), tf.cast(0.0, dtype=tf.float32), dtype=tf.float32,  name='action_one_hot')
        replay_chosen_q = tf.reduce_sum(self.q_values * replay_action_one_hot, reduction_indices=1, name='replay_chosen_q')
        dataset_expec = tf.reduce_mean(replay_chosen_q)
        negative_sampling = tf.reduce_mean(tf.reduce_logsumexp(self.q_values, 1))
        min_q_loss = self.min_q_weight * (negative_sampling - dataset_expec)
        self.regularizations.append(tf.cast(min_q_loss, tf.float32))

    def __str__(self):
        result = [
            "Dense (num outputs = {})".format(self.num_actions)
        ]
        return '\n'.join(result)

    def add_softmax_with_temperature(self):
        temperature = self.ap.network_wrappers[self.network_name].softmax_temperature
        temperature_scaled_outputs = self.q_values / temperature
        return tf.nn.softmax(temperature_scaled_outputs, name="softmax")

