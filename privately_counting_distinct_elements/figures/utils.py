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

"""Utils for generating figures."""

from typing import Iterable, Sequence


def print_table(file, values: Iterable[tuple[float, float]]) -> None:
    """Print plot table."""
    print("table {%", file=file)
    for value in values:
        print(value[0], value[1], file=file)
    print("};", file=file)


def print_shadow(file, values: Sequence[tuple[float, dict[str, float]]]) -> None:
    """Print plot table."""
    print(f"(axis cs:{values[0][0]},{values[0][1]['upper_bound']})--", file=file)
    for value in values:
        print(f"(axis cs:{value[0]},{value[1]['lower_bound']})--", file=file)
    for value in reversed(values):
        print(f"(axis cs:{value[0]},{value[1]['upper_bound']})--", file=file)
    print("cycle;", file=file)
