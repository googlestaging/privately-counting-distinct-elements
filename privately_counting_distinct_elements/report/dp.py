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

"""Utilities for computing differentially private estimates of distinct count."""

from typing import Callable, Sequence

import cachetools
import numpy as np

from privately_counting_distinct_elements import dataset, selection
from privately_counting_distinct_elements.report import common


def _selector_cache_key(
    epsilon: float,
    beta: float,
    counts: list[float],
    user_contribution_bounds: list[int],
) -> tuple[int, int, int, int]:
    """Returns a cache key for parameters of an algorithm selecting contribution bound."""
    return (
        int(epsilon * 1000),
        int(beta * 1000),
        id(counts),
        id(user_contribution_bounds),
    )


def _max_index(list_: list[float]) -> int:
    max_value = list_[0]
    max_index = 0
    for i in range(len(list_)):
        if list_[i] > max_value:
            max_value = list_[i]
            max_index = i
    return max_index


def _is_sorted(list_: list[float]) -> bool:
    for i in range(1, len(list_)):
        if list_[i] < list_[i - 1]:
            return False
    return True


def _is_concave(list_: list[float]) -> bool:
    max_index = _max_index(list_)
    for i in range(1, max_index + 1):
        if list_[i] < list_[i - 1]:
            return False
    for i in range(max_index + 1, len(list_)):
        if list_[i] > list_[i - 1]:
            return False
    return True


@cachetools.cached(cache={}, key=_selector_cache_key)
def _dp_max_utility_selector(
    epsilon: float,
    beta: float,
    counts: Sequence[float],
    user_contribution_bounds: Sequence[int],
) -> Callable[[], int]:
    """Select the best user contribution bound via epsislon-DP mechanism.

    Note that the resulting funciton is cached and values of epsilon and beta
    are rounded to the third significant digit.

    Args:
      epsilon: the privacy parameter used by the mechanism.
      beta: error probability used by the mechanism.
      counts: potential distinct counts that we are selecting from.
      user_contribution_bounds: contribution bounds used to generate the counts.

    Returns:
      A function samping estimated best user contribution bound.
    """
    utility = np.array(counts) - np.array(user_contribution_bounds) / epsilon * np.log(
        0.5 / beta
    )
    mechanism = selection.GeneralizedExponential(
        epsilon, beta, user_contribution_bounds, utility
    )

    def internal():
        return user_contribution_bounds[mechanism.randomise()]

    return internal


def greedy_distinct_count(
    data: dataset.DataSet,
    epsilon: float,
    user_contribution_bounds: list[int],
    beta: float,
) -> common.BoundWithCount:
    """Estimate the number of distinct elements using greedy algorithm."""
    return common.select_optimal(
        _dp_max_utility_selector(
            epsilon=epsilon / 2,
            beta=beta,
            counts=common.greedy_distinct_counts(data, user_contribution_bounds),
            user_contribution_bounds=user_contribution_bounds,
        ),
        lambda bound: common.dp_greedy_distinct_count(  # pylint: disable=g-long-lambda
            data,
            bound,
            epsilon / 2,
        )
        - bound / epsilon * np.log(0.5 / beta),
    )


def matching_distinct_count(
    data: dataset.DataSet,
    epsilon: float,
    user_contribution_bounds: list[int],
    beta: float,
) -> common.BoundWithCount:
    """Estimate the number of distinct elements using matching based algorithm."""
    return common.select_optimal(
        _dp_max_utility_selector(
            epsilon=epsilon / 2,
            beta=beta,
            counts=common.matching_distinct_counts(data, user_contribution_bounds),
            user_contribution_bounds=user_contribution_bounds,
        ),
        lambda bound: common.dp_matching_distinct_count(  # pylint: disable=g-long-lambda
            data,
            bound,
            epsilon / 2,
        )
        - bound / epsilon * np.log(0.5 / beta),
    )


def flow_distinct_count(
    data: dataset.DataSet,
    epsilon: float,
    user_contribution_bounds: list[int],
    beta: float,
) -> common.BoundWithCount:
    """Estimate the number of distinct elements using flow based algorithm."""
    return common.select_optimal(
        _dp_max_utility_selector(
            epsilon=epsilon / 2,
            beta=beta,
            counts=common.flow_distinct_counts(data, user_contribution_bounds),
            user_contribution_bounds=user_contribution_bounds,
        ),
        lambda bound: common.dp_flow_distinct_count(  # pylint: disable=g-long-lambda
            data,
            bound,
            epsilon / 2,
        )
        - bound / epsilon * np.log(0.5 / beta),
    )


def sampling_distinct_count(
    data: dataset.DataSet,
    epsilon: float,
    user_contribution_bounds: list[int],
    beta: float,
) -> common.BoundWithCount:
    """Estimate the number of distinct elements using sampling algorithm."""
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
        lambda bound: common.dp_sampling_distinct_count(  # pylint: disable=g-long-lambda
            data,
            bound,
            epsilon / 2,
        )
        - bound / epsilon * np.log(0.5 / beta),
    )
