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

from typing import Type

from rl_coach.base_parameters import NetworkComponentParameters
from rl_coach.architectures.head_parameters import HeadParameters


class ConservativeQuantileRegressionQHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='conservative_quantile_regression_q_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None, output_bias_initializer=None, min_q_weight=0.0):
        super().__init__(parameterized_class_name="ConservativeQuantileRegressionQHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)
        self.output_bias_initializer = output_bias_initializer
        self.min_q_weight = min_q_weight


class ConservativeQHeadParameters(HeadParameters):
    def __init__(self, activation_function: str ='relu', name: str='conservative_q_head_params',
                 num_output_head_copies: int = 1, rescale_gradient_from_head_by_factor: float = 1.0,
                 loss_weight: float = 1.0, dense_layer=None, output_bias_initializer=None, min_q_weight=0.0):
        super().__init__(parameterized_class_name="ConservativeQHead", activation_function=activation_function, name=name,
                         dense_layer=dense_layer, num_output_head_copies=num_output_head_copies,
                         rescale_gradient_from_head_by_factor=rescale_gradient_from_head_by_factor,
                         loss_weight=loss_weight)
        self.output_bias_initializer = output_bias_initializer
        self.min_q_weight = min_q_weight