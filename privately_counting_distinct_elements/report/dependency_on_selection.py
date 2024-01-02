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

"""Estimations of dependency of accuracy on the method selecting optimal bound."""

import dataclasses
from typing import Any, Callable

import numpy as np

from privately_counting_distinct_elements import dataset
from privately_counting_distinct_elements.report import (
    common,
    dp,
    half_dp,
    max_utility,
    quantile,
    stats,
)


@dataclasses.dataclass(frozen=True)
class BoundWithCountDistributionStats:
    """Stats of the distributions of bounds and counts."""

    bound: stats.DistributionStats
    count: stats.DistributionStats

    def as_json_encodable(self) -> dict[str, Any]:
        return dict(
            bound=self.bound.as_json_encodable(),
            count=self.count.as_json_encodable(),
        )


@dataclasses.dataclass(frozen=True)
class FixedSelectionReport:
    """Report of accuracies for some selction of optimal bound menthod."""

    sampling: BoundWithCountDistributionStats
    greedy: BoundWithCountDistributionStats
    matching: BoundWithCountDistributionStats

    def as_json_encodable(self) -> dict[str, Any]:
        return dict(
            sampling=self.sampling.as_json_encodable(),
            greedy=self.greedy.as_json_encodable(),
            matching=self.matching.as_json_encodable(),
        )


@dataclasses.dataclass(frozen=True)
class DependencyOnSelectionReport:
    """Report of the dependency of accuracy on selction of optimal bound menthod."""

    max_utility: FixedSelectionReport
    dp_max_utility_non_dp_estimation: FixedSelectionReport
    dp_max_utility_dp_estimation: FixedSelectionReport
    quantile_dp_estimation: FixedSelectionReport

    def as_json_encodable(self) -> dict[str, Any]:
        return dict(
            max_utility=self.max_utility.as_json_encodable(),
            dp_max_utility_non_dp_estimation=self.dp_max_utility_non_dp_estimation.as_json_encodable(),
            dp_max_utility_dp_estimation=self.dp_max_utility_dp_estimation.as_json_encodable(),
            quantile_dp_estimation=self.quantile_dp_estimation.as_json_encodable(),
        )


def _compute_stats(
    sampler: Callable[[], common.BoundWithCount],
    num_repetition: int,
) -> BoundWithCountDistributionStats:
    """Compute stats of the bounds and counts distributions."""
    data = [sampler() for _ in range(num_repetition)]

    bounds = [item.bound for item in data]
    counts = [item.count for item in data]

    return BoundWithCountDistributionStats(
        bound=stats.DistributionStats(
            np.quantile(bounds, 0.1),
            np.quantile(bounds, 0.5),
            np.quantile(bounds, 0.9),
        ),
        count=stats.DistributionStats(
            np.quantile(counts, 0.1),
            np.quantile(counts, 0.5),
            np.quantile(counts, 0.9),
        ),
    )


def _max_utility_report(
    data: dataset.DataSet,
    epsilon: float,
    contribution_bounds: list[int],
    num_repetition: int,
    beta: float,
) -> FixedSelectionReport:
    return FixedSelectionReport(
        sampling=_compute_stats(
            lambda: max_utility.sampling_distinct_count(  # pylint: disable=g-long-lambda
                data, epsilon, contribution_bounds, beta
            ),
            num_repetition,
        ),
        greedy=_compute_stats(
            lambda: max_utility.greedy_distinct_count(  # pylint: disable=g-long-lambda
                data, epsilon, contribution_bounds, beta
            ),
            num_repetition,
        ),
        matching=_compute_stats(
            lambda: max_utility.matching_distinct_count(  # pylint: disable=g-long-lambda
                data, epsilon, contribution_bounds, beta
            ),
            num_repetition,
        ),
    )


def _dp_max_utility_non_dp_estimation_report(
    data: dataset.DataSet,
    epsilon: float,
    contribution_bounds: list[int],
    num_repetition: int,
    beta: float,
) -> FixedSelectionReport:
    return FixedSelectionReport(
        sampling=_compute_stats(
            lambda: half_dp.sampling_distinct_count(  # pylint: disable=g-long-lambda
                data, epsilon, contribution_bounds, beta
            ),
            num_repetition,
        ),
        greedy=_compute_stats(
            lambda: half_dp.greedy_distinct_count(  # pylint: disable=g-long-lambda
                data, epsilon, contribution_bounds, beta
            ),
            num_repetition,
        ),
        matching=_compute_stats(
            lambda: half_dp.matching_distinct_count(  # pylint: disable=g-long-lambda
                data, epsilon, contribution_bounds, beta
            ),
            num_repetition,
        ),
    )


def _quantile_dp_estimation_report(
    data: dataset.DataSet,
    epsilon: float,
    num_repetition: int,
    beta: float,
) -> FixedSelectionReport:
    return FixedSelectionReport(
        sampling=_compute_stats(
            lambda: quantile.sampling_distinct_count(data, epsilon, beta),
            num_repetition,
        ),
        greedy=_compute_stats(
            lambda: quantile.greedy_distinct_count(data, epsilon, beta),
            num_repetition,
        ),
        matching=_compute_stats(
            lambda: quantile.matching_distinct_count(data, epsilon, beta),
            num_repetition,
        ),
    )


def _dp_max_utility_dp_estimation_report(
    data: dataset.DataSet,
    epsilon: float,
    contribution_bounds: list[int],
    num_repetition: int,
    beta: float,
) -> FixedSelectionReport:
    return FixedSelectionReport(
        sampling=_compute_stats(
            lambda: dp.sampling_distinct_count(  # pylint: disable=g-long-lambda
                data, epsilon, contribution_bounds, beta
            ),
            num_repetition,
        ),
        greedy=_compute_stats(
            lambda: dp.greedy_distinct_count(  # pylint: disable=g-long-lambda
                data, epsilon, contribution_bounds, beta
            ),
            num_repetition,
        ),
        matching=_compute_stats(
            lambda: dp.matching_distinct_count(  # pylint: disable=g-long-lambda
                data, epsilon, contribution_bounds, beta
            ),
            num_repetition,
        ),
    )


def dependency_on_selection_report(
    data: dataset.DataSet,
    epsilon: float,
    num_repetition: int,
    contribution_bounds: list[int],
    beta: float,
) -> DependencyOnSelectionReport:
    """Compute report for dependency of accuracy on bound selection menthod."""
    return DependencyOnSelectionReport(
        _max_utility_report(
            data,
            epsilon,
            contribution_bounds,
            num_repetition,
            beta,
        ),
        _dp_max_utility_non_dp_estimation_report(
            data,
            epsilon,
            contribution_bounds,
            num_repetition,
            beta,
        ),
        _dp_max_utility_dp_estimation_report(
            data,
            epsilon,
            contribution_bounds,
            num_repetition,
            beta,
        ),
        _quantile_dp_estimation_report(
            data,
            epsilon,
            num_repetition,
            beta,
        ),
    )
