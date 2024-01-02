# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for computing (non-private) estimates of distinct count.

In each of the algorithms, the optimal bound is estimated by using a
non-private 90th percentile of the dataset.
"""
import numpy as np

from privately_counting_distinct_elements import dataset
from privately_counting_distinct_elements.report import common


def _optimal_bound(data: dataset.DataSet) -> int:
    return int(np.quantile(data.as_matrix().sum(axis=1).getA1(), 0.9))


def greedy_distinct_count(
    data: dataset.DataSet,
    epsilon: float,
    beta: float,
) -> common.BoundWithCount:
    bound = _optimal_bound(data)
    return common.BoundWithCount(
        bound,
        common.dp_greedy_distinct_count(data, bound, epsilon)
        - bound / epsilon * np.log(0.5 / beta),
    )


def matching_distinct_count(
    data: dataset.DataSet,
    epsilon: float,
    beta: float,
) -> common.BoundWithCount:
    bound = _optimal_bound(data)
    return common.BoundWithCount(
        bound,
        common.dp_matching_distinct_count(data, bound, epsilon)
        - bound / epsilon * np.log(0.5 / beta),
    )


def sampling_distinct_count(
    data: dataset.DataSet,
    epsilon: float,
    beta: float,
) -> common.BoundWithCount:
    bound = _optimal_bound(data)
    return common.BoundWithCount(
        bound,
        common.dp_sampling_distinct_count(data, bound, epsilon)
        - bound / epsilon * np.log(0.5 / beta),
    )
