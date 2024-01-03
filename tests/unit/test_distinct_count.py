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

"""Test algorithms for computing numbers of distinct elements."""
from privately_counting_distinct_elements import dataset, distinct_count


def test_true_distinct_count() -> None:
    data = dataset.DataSet([[0, 1], [2, 1]])

    assert distinct_count.true_distinct_count(data) == 3


def test_greedy_distinct_count_1() -> None:
    data = dataset.DataSet([[0, 1], [2, 1]])

    assert distinct_count.greedy_distinct_count(data, 1) == 2


def test_greedy_distinct_count_2() -> None:
    data = dataset.DataSet([[0, 1], [2, 1]])

    assert distinct_count.greedy_distinct_count(data, 2) == 3


def test_greedy_distinct_count_3() -> None:
    data = dataset.DataSet([[1, 0, 2], [1, 3]])

    assert (
        distinct_count.greedy_distinct_count(data, 2) == 3
    ), distinct_count.greedy_distinct_count(data, 2)


def test_matching_distinct_count() -> None:
    data = dataset.DataSet([[0, 1, 2], [2, 1, 3]])

    assert (
        distinct_count.matching_distinct_count(data, 2) == 4
    ), distinct_count.matching_distinct_count(data, 2)


def test_flow_distinct_count() -> None:
    data = dataset.DataSet([[0, 1, 2], [2, 1, 3]])

    assert (
        distinct_count.flow_distinct_count(data, 2) == 4
    ), distinct_count.flow_distinct_count(data, 2)
