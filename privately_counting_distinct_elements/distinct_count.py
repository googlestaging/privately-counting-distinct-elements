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

"""Functions for computing distinct counts."""

import random

from scipy import sparse
from scipy.sparse import csgraph

from privately_counting_distinct_elements import dataset


def true_distinct_count(data: dataset.DataSet) -> int:
    """Compute the number of distinct elements.

    Args:
      data: the input dataset.

    Returns:
      the number of elements in the dataset.
    """
    return data.number_of_values


def _sample(values: list[int], num: int) -> list[int]:
    if len(values) < num:
        return values
    return random.sample(values, k=num)


def greedy_distinct_count(
    data: dataset.DataSet,
    user_contribution_bound: int,
) -> int:
    """Estimate number of distinct elements using greedy algorithm.

    Note that the algorithm is deterministic and it goes over the users in the
    order of their ids.

    Args:
      data: the input dataset.
      user_contribution_bound: bound on the contributions of a single user.

    Returns:
      the estimate of the number of elements in the dataset.
    """
    result = set()
    positions = [0] * len(data)
    for _ in range(user_contribution_bound):
        for i, r in enumerate(data):
            while positions[i] < len(r):
                index = r[positions[i]]
                positions[i] += 1
                if index not in result:
                    result.add(index)
                    break
    return len(result)


def matching_distinct_count(
    data: dataset.DataSet,
    user_contribution_bound: int,
) -> int:
    """Estimate number of distinct elements using matching-based algorithm.

    Args:
      data: the input dataset.
      user_contribution_bound: bound on the contributions of a single user.

    Returns:
      the estimate of the number of elements in the dataset.
    """
    return (
        csgraph.maximum_bipartite_matching(
            sparse.vstack([data.as_matrix()] * user_contribution_bound)
        )
        != -1
    ).sum()


def flow_distinct_count(
    data: dataset.DataSet,
    user_contribution_bound: int,
) -> int:
    """Estimate number of distinct elements using flow-based algorithm.

    Args:
      data: the input dataset.
      user_contribution_bound: bound on the contributions of a single user.

    Returns:
      the estimate of the number of elements in the dataset.
    """
    return csgraph.maximum_flow(
        data.as_flow_matrix(user_contribution_bound), 0, 1
    ).flow_value


def sampling_distinct_count(
    data: dataset.DataSet,
    user_contribution_bound: int,
) -> int:
    """Estimate number of distinct elements using sampling algorithm.

    Note that the algorithm is nondeterministic.

    Args:
      data: the input dataset.
      user_contribution_bound: bound on the contributions of a single user.

    Returns:
      the estimate of the number of elements in the dataset.
    """
    result = set()
    for record in data:
        result.update(_sample(record, user_contribution_bound))
    return len(result)
