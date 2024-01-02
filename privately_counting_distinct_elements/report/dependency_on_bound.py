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

"""Utilities to create report for dependency on the contribution bound."""

import dataclasses
from typing import Any, Iterable, List

from rich import progress

from privately_counting_distinct_elements import dataset
from privately_counting_distinct_elements.report import common, stats


@dataclasses.dataclass
class DependencyOnContributionBoundReportItem:
    """Report containing estimations for all algorithms for a fixed bound."""

    contribution_bound: int
    counts: common.CountsReport

    def as_json_encodable(self) -> dict[str, Any]:
        return dict(
            contribution_bound=self.contribution_bound,
            counts=self.counts.as_json_encodable(),
        )


@dataclasses.dataclass
class DependencyOnContributionBoundReport:
    """Report containing estimations for all algorithms for different bounds."""

    items: List[DependencyOnContributionBoundReportItem]

    def as_json_encodable(self) -> list[dict[str, Any]]:
        return [item.as_json_encodable() for item in self.items]


def dependency_on_contribution_bound_report_item(
    data: dataset.DataSet,
    contribution_bound: int,
    num_repetition: int,
) -> DependencyOnContributionBoundReportItem:
    """Compute report for given contribution bound."""
    return DependencyOnContributionBoundReportItem(
        contribution_bound,
        common.CountsReport(
            matching=stats.compute(
                lambda: common.matching_distinct_count(data, contribution_bound),
                1,
            ),
            greedy=stats.compute(
                lambda: common.greedy_distinct_count(data, contribution_bound),
                1,
            ),
            sampling=stats.compute(
                lambda: common.sampling_distinct_count(data, contribution_bound),
                num_repetition,
            ),
        ),
    )


def dependency_on_contribution_bound_report(
    data: dataset.DataSet,
    contribution_bounds: Iterable[int],
    num_repetition: int,
) -> DependencyOnContributionBoundReport:
    """Compute report for given contribution bounds."""
    items = []
    with progress.Progress(transient=True) as progress_bar:
        for contribution_bound in progress_bar.track(contribution_bounds):
            items.append(
                dependency_on_contribution_bound_report_item(
                    data,
                    contribution_bound,
                    num_repetition,
                )
            )
    return DependencyOnContributionBoundReport(items)
