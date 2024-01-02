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

"""Define utility functions for loading data sets."""

import json
import logging
import pathlib
from typing import Iterable, Iterator

import nltk
from scipy import sparse


def _to_matrix(data: Iterable[list[int]]) -> sparse.csr_matrix:
    """Convert dataset to the biadjacency matrix of the corresponding graph."""
    row_ind = []
    col_ind = []
    for index, row in enumerate(data):
        for item in row:
            row_ind.append(index)
            col_ind.append(item)
    return sparse.csr_matrix(([1] * len(row_ind), (row_ind, col_ind)))


def _to_flow_matrix(
    number_of_values: int,
    data: list[list[int]],
    user_contribution_bound: int,
) -> sparse.csr_matrix:
    """Convert dataset to the flow matrix of the corresponding graph."""
    row_ind = []
    col_ind = []
    number_of_users = len(data)
    # for each user we add an edge from source
    for i in range(2, number_of_users + 2):
        row_ind.append(0)
        col_ind.append(i)
    # for each value we add an edge to sink
    for i in range(number_of_users + 2, number_of_values + number_of_users + 2):
        row_ind.append(i)
        col_ind.append(1)
    number_of_internal_edges = 0
    # for each user and each value we have an edge
    for index, row in enumerate(data):
        for item in row:
            # we shift by 2 since we have source and sink
            row_ind.append(index + 2)
            # we shift by number_of_users + 2 since we have source and sink
            col_ind.append(item + number_of_users + 2)
            number_of_internal_edges += 1
    weights = (
        # all edges from source have caapcity of user_contribution
        ([user_contribution_bound] * number_of_users)
        # each edge from a value to sink has capacity one
        + ([1] * number_of_values)
        # from each user to each value this user has we have an edge of weifht 1
        + ([1] * number_of_internal_edges)
    )
    return sparse.csr_matrix((weights, (row_ind, col_ind)))


class DataSet:
    """Data set for computing distinct count and set union."""

    def __init__(self, data: list[list[int]]):
        self._data = data
        self._matrix = None
        self._degree = max(len(contribution) for contribution in self._data)
        self.number_of_values = len(
            {value for record in self._data for value in record}
        )
        self._flow_matrix: dict[int, sparse.csr_matrix] = {}
        self.number_of_records = sum(len(record) for record in self._data)

    def as_matrix(self) -> sparse.csr_matrix:
        if self._matrix is None:
            self._matrix = _to_matrix(self._data)
        return self._matrix

    def as_flow_matrix(self, user_contribution_bound: int) -> sparse.csr_matrix:
        if user_contribution_bound not in self._flow_matrix:
            self._flow_matrix[user_contribution_bound] = _to_flow_matrix(
                self.number_of_values, self._data, user_contribution_bound
            )
        return self._flow_matrix[user_contribution_bound]

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def degree(self) -> int:
        return self._degree

    @property
    def number_of_users(self) -> int:
        return len(self._data)


def _split_text(text):
    """Split given text into words and ignore punctuation."""
    words = nltk.word_tokenize(text)
    return set(word.lower() for word in words if word.isalpha() or word.isalnum())


def prepare_dataset(path: pathlib.Path) -> DataSet:
    """Read the data set from the given path."""
    word_matching: dict[str, int] = {}
    user_matching: dict[str, int] = {}

    result: list[set[int]] = []

    with path.open("r") as f:
        for line in f:
            try:
                data = json.loads(line)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning(
                    "Failed to parse: %s; caused by %s",
                    line,
                    e,
                )
                continue

            if "reviewText" not in data:
                continue
            try:
                split = _split_text(data["reviewText"])
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning(
                    "Failed to split: %s; caused by %s",
                    data["reviewText"],
                    e,
                )
                continue
            user_id = data["reviewerID"]
            for word in split:
                if word not in word_matching:
                    word_matching[word] = len(word_matching)
                if user_matching.get(user_id) is None:
                    result.append(set())
                    user_matching[user_id] = len(user_matching)
                result[user_matching[user_id]].add(word_matching[word])
    return DataSet([list(record) for record in result])
