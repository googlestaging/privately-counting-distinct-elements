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

"""Utilities to create report for dependency on epsilon."""

import dataclasses
from typing import Any, Iterable, List

from rich import progress

from privately_counting_distinct_elements import dataset, shifted_inverse
from privately_counting_distinct_elements.report import common, dp, stats

ERROR_LEVEL = 0.001


@dataclasses.dataclass
class DependencyOnEpsilonReportItem:
    """Report containing estimations for all algorithms for a fixed bound."""

    epsilon: float
    counts: common.CountsReport

    def as_json_encodable(self) -> dict[str, Any]:
        return dict(
            epsilon=self.epsilon,
            counts=self.counts.as_json_encodable(),
        )


@dataclasses.dataclass
class DependencyOnEpsilonReport:
    """Report containing estimations for all algorithms for several bounds."""

    items: List[DependencyOnEpsilonReportItem]

    def as_json_encodable(self) -> list[dict[str, Any]]:
        return [item.as_json_encodable() for item in self.items]


def dependency_on_epsilon_report_item(
    data: dataset.DataSet,
    epsilon: float,
    num_repetition: int,
    contribution_bounds: list[int],
    beta: float,
) -> DependencyOnEpsilonReportItem:
    """Compute report for a given contribution bound."""
    return DependencyOnEpsilonReportItem(
        epsilon,
        common.CountsReport(
            matching=stats.compute(
                lambda: common.ignore_bound(  # pylint: disable=g-long-lambda
                    dp.matching_distinct_count(data, epsilon, contribution_bounds, beta)
                ),
                num_repetition,
            ),
            greedy=stats.compute(
                lambda: common.ignore_bound(  # pylint: disable=g-long-lambda
                    dp.greedy_distinct_count(data, epsilon, contribution_bounds, beta)
                ),
                num_repetition,
            ),
            sampling=stats.compute(
                lambda: common.ignore_bound(  # pylint: disable=g-long-lambda
                    dp.sampling_distinct_count(data, epsilon, contribution_bounds, beta)
                ),
                num_repetition,
            ),
            shifted_inverse=stats.compute(
                lambda: shifted_inverse.shifted_inverse_distinct_count(  # pylint: disable=g-long-lambda
                    data, epsilon, ERROR_LEVEL, data.number_of_values, beta
                ),
                num_repetition,
            ),
        ),
    )


def dependency_on_epsilon_report(
    data: dataset.DataSet,
    epsilons: Iterable[float],
    num_repetition: int,
    contribution_bounds: list[int],
    beta: float,
) -> DependencyOnEpsilonReport:
    """Compute report for given contribution bounds."""
    items = []
    with progress.Progress(transient=True) as progress_bar:
        for epsilon in progress_bar.track(epsilons):
            items.append(
                dependency_on_epsilon_report_item(
                    data,
                    epsilon,
                    num_repetition,
                    contribution_bounds,
                    beta,
                )
            )
    return DependencyOnEpsilonReport(items)
