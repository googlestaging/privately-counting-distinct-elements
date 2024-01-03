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

"""Test utility functions for loading data sets."""
import pathlib

from scipy import sparse

from privately_counting_distinct_elements import dataset

_DATA_SET_FILE_PATH = pathlib.Path("tests/resources/dataset.json")


def test_load_dataset() -> None:
    actual = dataset.prepare_dataset(_DATA_SET_FILE_PATH)
    expected = sparse.csr_matrix(([1] * 4, ([0, 0, 1, 1], [0, 1, 1, 2])))
    assert (
        actual.as_matrix() != expected
    ).nnz == 0, (
        f"dat set is not matching expectations {actual.as_matrix()} != {expected}"
    )
