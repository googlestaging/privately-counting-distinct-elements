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

"""Utilities for computing half-private estimates of distinct count.

These utilities are computing estimates of the number of distinct elements
by selecting user-contribution bounds non-privately and estimating the count
privately.
"""

from typing import Callable, Sequence

import numpy as np

from privately_counting_distinct_elements import dataset, selection
from privately_counting_distinct_elements.report import common


def _dp_max_utility_selector(
    epsilon: float,
    beta: float,
    counts: Sequence[float],
    user_contribution_bounds: Sequence[int],
) -> Callable[[], int]:
    """Select the best user contribution bound via epsislon-DP mechanism."""

    def internal():
        utility = np.array(counts) - np.array(
            user_contribution_bounds
        ) / epsilon * np.log(0.5 / 0.001)

        return user_contribution_bounds[
            selection.GeneralizedExponential(
                epsilon, beta, user_contribution_bounds, utility
            ).randomise()
        ]

    return internal


def greedy_distinct_count(
    data: dataset.DataSet,
    epsilon: float,
    user_contribution_bounds: list[int],
    beta: float,
) -> common.BoundWithCount:
    return common.select_optimal(
        _dp_max_utility_selector(
            epsilon=epsilon / 2,
            beta=beta,
            counts=common.greedy_distinct_counts(data, user_contribution_bounds),
            user_contribution_bounds=user_contribution_bounds,
        ),
        lambda bound: common.greedy_distinct_count(data, bound),
    )


def matching_distinct_count(
    data: dataset.DataSet,
    epsilon: float,
    user_contribution_bounds: list[int],
    beta: float,
) -> common.BoundWithCount:
    return common.select_optimal(
        _dp_max_utility_selector(
            epsilon=epsilon / 2,
            beta=beta,
            counts=common.matching_distinct_counts(data, user_contribution_bounds),
            user_contribution_bounds=user_contribution_bounds,
        ),
        lambda bound: common.matching_distinct_count(data, bound),
    )


def sampling_distinct_count(
    data: dataset.DataSet,
    epsilon: float,
    user_contribution_bounds: list[int],
    beta: float,
) -> common.BoundWithCount:
    counts = [
        common.sampling_distinct_count(data, bound)
        for bound in user_contribution_bounds
    ]
    return common.select_optimal(
        _dp_max_utility_selector(
            epsilon=epsilon / 2,
            beta=beta,
            counts=counts,
            user_contribution_bounds=user_contribution_bounds,
        ),
        lambda bound: common.sampling_distinct_count(data, bound),
    )
