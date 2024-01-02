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

"""Utilities for computing stats of a distribution."""

import dataclasses
from typing import Any, Callable

import numpy as np


@dataclasses.dataclass(frozen=True)
class DistributionStats:
    """Statistics about a distribution of real numbers."""

    upper_bound: np.floating[Any]
    median: np.floating[Any]
    lower_bound: np.floating[Any]

    def as_json_encodable(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def compute(sampler: Callable[[], float], num_repetition: int) -> DistributionStats:
    data = [sampler() for _ in range(num_repetition)]
    return DistributionStats(
        np.quantile(data, 0.1),
        np.quantile(data, 0.5),
        np.quantile(data, 0.9),
    )
