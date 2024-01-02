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

"""Implementation of the shifted inverse mechanism for count distinct.

This implementation follows https://dl.acm.org/doi/abs/10.1145/3548606.3560567
and https://github.com/User225872/ShiftedInverse/.
"""
import math

import cachetools
import numpy as np
from diffprivlib import mechanisms
from scipy import optimize, sparse

from privately_counting_distinct_elements import dataset


@cachetools.cached(cache={}, key=lambda data: id(data))
def _bounds(data: dataset.DataSet) -> list[tuple[int, int]]:
    return [(0, 1) for _ in range(data.number_of_users + data.number_of_values)]


@cachetools.cached(cache={}, key=lambda data: id(data))
def _objective_coefficients(data: dataset.DataSet) -> np.ndarray:
    return -np.concatenate(
        [np.zeros(data.number_of_users), np.ones(data.number_of_values)]
    )


@cachetools.cached(cache={}, key=lambda data: id(data))
def _inequalities_coefficients(data: dataset.DataSet) -> sparse.csr_matrix:
    """Build the left side of the inequalities for the optimization problem."""
    row_ids = []
    col_ids = []
    val = []

    row_id = 0

    for user_id, records in enumerate(data):
        for record in records:
            # we build left side of the inequality `w_r <= w_u`
            row_ids.append(row_id)
            col_ids.append(data.number_of_users + record)
            val.append(1)
            row_ids.append(row_id)
            col_ids.append(user_id)
            val.append(-1)

            row_id += 1

    assert row_id == data.number_of_records

    # we build the left side of the inequality sum(w_u) <= j
    for user_id in range(data.number_of_users):
        row_ids.append(row_id)
        col_ids.append(user_id)
        val.append(1)

    return sparse.csr_matrix((val, (row_ids, col_ids)))


@cachetools.cached(cache={}, key=lambda data: id(data))
def _inequalities_bounds(data: dataset.DataSet) -> np.ndarray:
    """Build the right side of the inequalities for the optimization problem.

    Note that the last bound is missing since it depends on the number of users
    we intend to delete.

    Args:
      data: the input dataset.

    Returns:
      a vector of right sides of all inrqualtiies except the one limiting the
      number of removed users.
    """
    return np.zeros(data.number_of_records)


@cachetools.cached(
    cache={}, key=lambda data, users_to_delete: (id(data), users_to_delete)
)
def _sensitivity(data: dataset.DataSet, users_to_delete: int) -> int:
    return -optimize.linprog(
        c=_objective_coefficients(data),
        bounds=_bounds(data),
        A_ub=_inequalities_coefficients(data),
        b_ub=np.concatenate([_inequalities_bounds(data), [users_to_delete]]),
    ).fun


def _round(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def shifted_inverse_distinct_count(
    data: dataset.DataSet,
    epsilon: float,
    error_level: float,
    upper_bound: int,
    beta: float,
) -> int:
    """Compute the number of distinct elements using Shifter Inverse."""
    log_upper_bound = math.log(upper_bound, 10)

    tau: int = math.ceil(
        2 / epsilon * math.log((log_upper_bound / error_level + 1) / beta)
    )

    results = [log_upper_bound]
    for users_to_delete in range(2 * tau):
        result_0 = _sensitivity(data, users_to_delete)
        result_1 = math.log(data.number_of_values - result_0, 10)
        result_2 = _round(result_1, error_level)
        results.append(result_2)

    utility: list[float] = [0] * (2 * tau)
    for i in range(0, tau):
        utility[i] = (-tau - 1 + i) * (results[i] - results[i + 1])
    for i in range(tau, 2 * tau):
        utility[i] = (tau - i) * (results[i] - results[i + 1])

    sampled = mechanisms.Exponential(
        epsilon=epsilon / 2,
        sensitivity=1,
        utility=utility,
        candidates=list(range(2 * tau)),
    ).randomise()

    return int(math.pow(10, np.random.uniform(results[sampled + 1], results[sampled])))
