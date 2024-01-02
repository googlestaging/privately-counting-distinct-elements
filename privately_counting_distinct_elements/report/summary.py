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

"""Binary generating reports for a given dataset."""

import json
import pathlib
from typing import Sequence

import rich
from absl import app, flags

from privately_counting_distinct_elements import dataset, distinct_count
from privately_counting_distinct_elements.report import (
    dependency_on_bound,
    dependency_on_epsilon,
    dependency_on_selection,
)

_DATA_SET = flags.DEFINE_string("data_set", None, "data set path", required=True)
_OUTPUT = flags.DEFINE_string("output", None, "output path", required=True)
_MAX_USER_CONTRIBUTION_BOUND = flags.DEFINE_integer(
    "max_user_contribution_bound", 100, "max user contribution bound"
)
_MAX_EPSILON = flags.DEFINE_integer("max_epsilon", 30, "max epsilon")
_EPSILON = flags.DEFINE_float("epsilon", 2.0, "epsilon")
_NUM_REPETITION = flags.DEFINE_integer(
    "num_repetition",
    100,
    "nummber of times a random variable is sampled to compute quantiles",
)
_BETA = flags.DEFINE_float("beta", 0.05, "beta")


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    data_set = dataset.prepare_dataset(pathlib.Path(_DATA_SET.value))
    rich.print("[green]SUCCESS:[/green] the dataset is loaded")
    user_contribution_bounds = list(range(1, _MAX_USER_CONTRIBUTION_BOUND.value))
    epsilons = list(range(1, _MAX_EPSILON.value))

    dependency_on_bound_report = (
        dependency_on_bound.dependency_on_contribution_bound_report(
            data_set,
            user_contribution_bounds,
            _NUM_REPETITION.value,
        ).as_json_encodable()
    )
    rich.print("[green]SUCCESS:[/green] dependency on bounds is estimated")

    dependency_on_epsilon_report = dependency_on_epsilon.dependency_on_epsilon_report(
        data_set,
        epsilons,
        _NUM_REPETITION.value,
        user_contribution_bounds,
        beta=_BETA.value,
    ).as_json_encodable()
    rich.print("[green]SUCCESS:[/green] dependency on epsilon is estimated")

    dependency_on_selection_report = (
        dependency_on_selection.dependency_on_selection_report(
            data_set,
            _EPSILON.value,
            _NUM_REPETITION.value,
            user_contribution_bounds,
            _BETA.value,
        ).as_json_encodable()
    )
    rich.print("[green]SUCCESS:[/green] dependency on selection is estimated")

    true_distinct_count = int(distinct_count.true_distinct_count(data_set))
    rich.print("[green]SUCCESS:[/green] true distinct count is computed")

    with pathlib.Path(_OUTPUT.value).open("w") as output_file:
        json.dump(
            dict(
                true_distinct_count=true_distinct_count,
                dependency_on_bound=dependency_on_bound_report,
                dependency_on_epsilon=dependency_on_epsilon_report,
                dependency_on_selection=dependency_on_selection_report,
            ),
            output_file,
        )


if __name__ == "__main__":
    app.run(main)
