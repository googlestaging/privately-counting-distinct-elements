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

"""A package defining DP mechanism for selection problem."""

import math
from typing import Sequence

from diffprivlib import mechanisms


class GeneralizedExponential:
    """A differentially private mechanism for selecting value with maximal utility.

    It assumes that sensitivity is increasing and utility is concave.
    """

    def __init__(
        self,
        epsilon: float,
        beta: float,
        sensitivity: Sequence[float] | float,
        utility: Sequence[float],
    ):
        if not isinstance(sensitivity, Sequence):
            sensitivity = [sensitivity] * len(utility)

        assert len(sensitivity) == len(utility), f"{len(sensitivity)} != {len(utility)}"
        assert 0 not in sensitivity

        t = 2 * math.log(len(utility) / beta) / epsilon
        min_utility = min(utility)
        utility = [u - min_utility for u in utility]
        reweighed_utility = []
        for i in range(len(utility)):
            reweighed_utility.append(
                -max(
                    (
                        (-utility[i] + sensitivity[i] * t)
                        - (-utility[j] + sensitivity[j] * t)
                    )
                    / (sensitivity[i] + sensitivity[j])
                    for j in range(len(utility))
                )
            )

        self.mechanism = mechanisms.Exponential(
            epsilon=epsilon, sensitivity=1, utility=reweighed_utility
        )

    def randomise(self) -> int:
        return self.mechanism.randomise()
