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

"""The binary generating plot showing dependency of accuracy on epsilon."""

import json
from typing import Sequence

from absl import app, flags

from privately_counting_distinct_elements.figures import utils

_REPORT = flags.DEFINE_string("report", None, "path to report", required=True)
_OUTPUT = flags.DEFINE_string("output", None, "output path", required=True)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    with open(_REPORT.value, "r") as file:
        report = json.load(file)

    true_count = [
        (item["epsilon"], report["true_distinct_count"])
        for item in report["dependency_on_epsilon"]
    ]
    matching = [
        (item["epsilon"], item["counts"]["matching"]["median"])
        for item in report["dependency_on_epsilon"]
    ]
    matching_bounds = [
        (item["epsilon"], item["counts"]["matching"])
        for item in report["dependency_on_epsilon"]
    ]
    greedy = [
        (item["epsilon"], item["counts"]["greedy"]["median"])
        for item in report["dependency_on_epsilon"]
    ]
    greedy_bounds = [
        (item["epsilon"], item["counts"]["greedy"])
        for item in report["dependency_on_epsilon"]
    ]

    sampling = [
        (item["epsilon"], item["counts"]["sampling"]["median"])
        for item in report["dependency_on_epsilon"]
    ]
    sampling_bounds = [
        (item["epsilon"], item["counts"]["sampling"])
        for item in report["dependency_on_epsilon"]
    ]

    shifted_inverse = [
        (item["epsilon"], item["counts"]["shifted_inverse"]["median"])
        for item in report["dependency_on_epsilon"]
    ]
    shifted_inverse_bounds = [
        (item["epsilon"], item["counts"]["shifted_inverse"])
        for item in report["dependency_on_epsilon"]
    ]

    with open(_OUTPUT.value, "w") as file:
        print("\\addplot [dashed, steelblue31119180]", file=file)
        utils.print_table(file, true_count)

        print("\\path [fill=forestgreen4416044, fill opacity=0.2]", file=file)
        utils.print_shadow(file, matching_bounds)

        print("\\addplot [densely dotted, thick, forestgreen4416044]", file=file)
        utils.print_table(file, matching)

        print("\\path [fill=darkorange25512714, fill opacity=0.2]", file=file)
        utils.print_shadow(file, greedy_bounds)

        print("\\addplot [densely dotted, thick, darkorange25512714]", file=file)
        utils.print_table(file, greedy)

        print("\\path [fill=crimson2143940, fill opacity=0.2]", file=file)
        utils.print_shadow(file, sampling_bounds)

        print("\\addplot [dashdotted, thick, crimson2143940]", file=file)
        utils.print_table(file, sampling)

        print("\\path [fill=steelblue31119180, fill opacity=0.2]", file=file)
        utils.print_shadow(file, shifted_inverse_bounds)

        print("\\addplot [dashdotted, thick, steelblue31119180]", file=file)
        utils.print_table(file, shifted_inverse)


if __name__ == "__main__":
    app.run(main)
