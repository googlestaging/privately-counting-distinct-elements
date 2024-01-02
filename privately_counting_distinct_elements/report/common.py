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

"""Package contains definitions of methods used for all parts of the report.

The difference between counting methids defined here and in distinct_count
package is that methods defined here are using memoization (cache) to
avoid computing the same value multiple times.
"""

import dataclasses
import functools
import random
from typing import Any, Callable, Hashable

import cachetools
from diffprivlib import mechanisms

from privately_counting_distinct_elements import dataset, distinct_count
from privately_counting_distinct_elements.report import stats


def _computation_cache_key(
    data: dataset.DataSet, user_contribution_bounds: list[int]
) -> tuple[int, int]:
    """Returns a cache key for parameters of an algorithm for distinct counts."""
    return (
        id(data),
        id(user_contribution_bounds),
    )


@dataclasses.dataclass
class CountsReport:
    """Report containing estimations for all algorithms."""

    matching: stats.DistributionStats | None = None
    greedy: stats.DistributionStats | None = None
    sampling: stats.DistributionStats | None = None
    shifted_inverse: stats.DistributionStats | None = None

    def as_json_encodable(self) -> dict[str, Any]:
        return dict(
            matching=self.matching.as_json_encodable() if self.matching else None,
            greedy=self.greedy.as_json_encodable() if self.greedy else None,
            sampling=self.sampling.as_json_encodable() if self.sampling else None,
            shifted_inverse=self.shifted_inverse.as_json_encodable()
            if self.shifted_inverse
            else None,
        )


@dataclasses.dataclass
class BoundWithCount:
    bound: int
    count: float


def select_optimal(
    selector: Callable[[], int],
    sampler: Callable[[int], float],
) -> BoundWithCount:
    optimal = selector()

    return BoundWithCount(optimal, sampler(optimal))


def ignore_bound(bound_with_count: BoundWithCount) -> float:
    return bound_with_count.count


@cachetools.cached(cache={}, key=lambda data, bound: (id(data), bound))
def matching_distinct_count(
    data: dataset.DataSet,
    user_contribution_bound: int,
) -> int:
    if user_contribution_bound > data.degree():
        return matching_distinct_count(data, data.degree())
    return distinct_count.matching_distinct_count(data, user_contribution_bound)


@cachetools.cached(cache={}, key=_computation_cache_key)
def matching_distinct_counts(
    data: dataset.DataSet,
    user_contribution_bounds: list[int],
) -> list[int]:
    """Computes matching distinct counts for each user contribution bound.

    Note that the function is caching the return value.

    Args:
      data: the input dataset.
      user_contribution_bounds: a list of bounds.

    Returns:
      a list with countds computed via matching distinct count algorithm.
    """
    return [matching_distinct_count(data, bound) for bound in user_contribution_bounds]


@cachetools.cached(cache={}, key=lambda data, bound: (id(data), bound))
def flow_distinct_count(
    data: dataset.DataSet,
    user_contribution_bound: int,
) -> int:
    if user_contribution_bound > data.degree():
        return flow_distinct_count(data, data.degree())
    return distinct_count.flow_distinct_count(data, user_contribution_bound)


@cachetools.cached(cache={}, key=_computation_cache_key)
def flow_distinct_counts(
    data: dataset.DataSet,
    user_contribution_bounds: list[int],
) -> list[int]:
    """Computes flow distinct counts for each user contribution bound.

    Note that the function is caching the return value.

    Args:
      data: the input dataset.
      user_contribution_bounds: a list of bounds.

    Returns:
      a list with counts computed via matching distinct count algorithm.
    """
    return [flow_distinct_count(data, bound) for bound in user_contribution_bounds]


@cachetools.cached(cache={}, key=lambda data, bound: (id(data), bound))
def greedy_distinct_count(
    data: dataset.DataSet,
    user_contribution_bound: int,
) -> int:
    if user_contribution_bound > data.degree():
        return greedy_distinct_count(data, data.degree())
    return distinct_count.greedy_distinct_count(data, user_contribution_bound)


@cachetools.cached(cache={}, key=_computation_cache_key)
def greedy_distinct_counts(
    data: dataset.DataSet,
    user_contribution_bounds: list[int],
) -> list[int]:
    """Computes greedy distinct counts for each user contribution bound.

    Note that the function is caching the return value.

    Args:
      data: the input dataset.
      user_contribution_bounds: a list of bounds.

    Returns:
      a list with counts computed via greedy distinct count algorithm.
    """
    return [greedy_distinct_count(data, bound) for bound in user_contribution_bounds]


def cached_nondeterministic(
    cache: dict[Hashable, list[Any]],
    key: Callable[..., Hashable],
    num_values: int = 100,
) -> Callable[[Any], Any]:
    """Cache several values of nondeterministic function."""

    def wrapper(func: Any) -> Any:
        @functools.wraps(func)
        def internal(*args, **kwargs) -> Any:
            key_value = key(*args, **kwargs)
            if key_value not in cache:
                value = func(*args, **kwargs)
                cache[key_value] = [value]
                return value

            if len(cache[key_value]) < num_values:
                value = func(*args, **kwargs)
                cache[key_value].append(value)
                return value
            return random.choice(cache[key_value])

        return internal

    return wrapper


@cached_nondeterministic(cache={}, key=lambda data, bound: (id(data), bound))
def sampling_distinct_count(
    data: dataset.DataSet,
    user_contribution_bound: int,
) -> int:
    return distinct_count.sampling_distinct_count(data, user_contribution_bound)


def dp_matching_distinct_count(
    data: dataset.DataSet,
    user_contribution_bound: int,
    epsilon: float,
) -> int:
    return mechanisms.Laplace(
        epsilon=epsilon, sensitivity=user_contribution_bound
    ).randomise(matching_distinct_count(data, user_contribution_bound))


def dp_flow_distinct_count(
    data: dataset.DataSet,
    user_contribution_bound: int,
    epsilon: float,
) -> int:
    return mechanisms.Laplace(
        epsilon=epsilon, sensitivity=user_contribution_bound
    ).randomise(flow_distinct_count(data, user_contribution_bound))


def dp_greedy_distinct_count(
    data: dataset.DataSet,
    user_contribution_bound: int,
    epsilon: float,
) -> float:
    return mechanisms.Laplace(
        epsilon=epsilon, sensitivity=user_contribution_bound
    ).randomise(greedy_distinct_count(data, user_contribution_bound))


def dp_sampling_distinct_count(
    data: dataset.DataSet,
    user_contribution_bound: int,
    epsilon: float,
) -> int:
    return mechanisms.Laplace(
        epsilon=epsilon, sensitivity=user_contribution_bound
    ).randomise(sampling_distinct_count(data, user_contribution_bound))
