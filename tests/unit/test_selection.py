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

"""Test selection mechanisms."""

import numpy as np

from privately_counting_distinct_elements import selection

ARRAY_LENGTH = 10
BETA = 0.1
LARGE_EPSILON = 1000


def test_large_epsilon_1():
    """Test that mechanism works when epsilon is large."""

    utility = np.random.rand(ARRAY_LENGTH)

    assert selection.GeneralizedExponential(
        epsilon=LARGE_EPSILON,
        beta=BETA,
        utility=utility,
        sensitivity=[1] * ARRAY_LENGTH,
    ).randomise() == np.argmax(utility)


def test_large_epsilon_2():
    """Test that mechanism works when epsilon is large."""

    utility = np.random.rand(ARRAY_LENGTH)

    assert selection.GeneralizedExponential(
        epsilon=LARGE_EPSILON, beta=BETA, utility=utility, sensitivity=1
    ).randomise() == np.argmax(utility)
