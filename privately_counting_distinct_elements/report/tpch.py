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

"""Estimate error of the matching and greedy algorithms on TPC-H dataset."""
from typing import Callable, Sequence

import pandas as pd
import rich
from absl import app, flags
from rich import progress

from privately_counting_distinct_elements import (
    dataset,
    distinct_count,
    shifted_inverse,
)
from privately_counting_distinct_elements.report import common, dp

_EPSILON = flags.DEFINE_float(
    "epsilon",
    1.0,
    "the value of the privacy parameter ",
    required=False,
)

_AVAIL_QTY = flags.DEFINE_string(
    "avail_qty", None, "avail qty data set path", required=True
)
_EXTENDED_PRICE = flags.DEFINE_string(
    "extended_price", None, "receipt date data set path", required=True
)
_ORDER_DATE = flags.DEFINE_string(
    "order_date", None, "order date data set path", required=True
)
_RECEIPT_DATE = flags.DEFINE_string(
    "receipt_date", None, "receipt date data set path", required=True
)


def _prepare_dataset(
    df: pd.DataFrame, *, user_column: str, value_column: str
) -> dataset.DataSet:
    """Prepares a dataset from a pandas dataframe.

    It assumes that the inpute dataframe has two columns `value_column` and
    `user_column`, where each row of the dataframe is a record.

    Args:
      df: the dataframe containing events.
      user_column: name of the column containing user ids.
      value_column: name of the column containing values corresponding to events.

    Returns:
      the user set where values are grouped by user id.
    """
    value_matching: dict[str, int] = {}
    user_matching: dict[str, int] = {}

    result: list[set[int]] = []

    for _, row in df.iterrows():
        if row[value_column] not in value_matching:
            value_matching[row[value_column]] = len(value_matching)
        if user_matching.get(row[user_column]) is None:
            result.append(set())
            user_matching[row[user_column]] = len(user_matching)
        result[user_matching[row[user_column]]].add(value_matching[row[value_column]])
    return dataset.DataSet([list(record) for record in result])


def _error(
    data: dataset.DataSet,
    dp_count: Callable[
        [dataset.DataSet, float, list[int], float], common.BoundWithCount
    ],
    progress_bar: progress.Progress,
    bounds: list[int],
) -> float:
    """Computes the average error of estimating distinct count.

    It computes the count 100 times, drop top 20 and bottom 20 values, and
    compute the average value for the rest.

    Args:
      data: the input dataset.
      dp_count: the algorithm computing distinct counts.
      progress_bar: rich progress bar used to show progress.
      bounds: bounds that needs to be considered by the algorithm.

    Returns:
      an average error of the algorithm.
    """
    counts = []
    for _ in progress_bar.track(range(100)):
        counts.append(dp_count(data, _EPSILON.value, bounds, 0.1).count)
    true_count = distinct_count.true_distinct_count(data)
    errors = [abs(count - true_count) for count in counts]
    errors.sort()
    return sum(errors[20:80]) / 60.0


def _shifted_inverse_error(
    data: dataset.DataSet,
    progress_bar: progress.Progress,
    bound: int,
) -> float:
    """Computes the average error of estimating distinct count.

    It computes the count 100 times, drop top 20 and bottom 20 values, and
    compute the average value for the rest.

    Args:
      data: the input dataset.
      progress_bar: rich progress bar used to show progress.
      bound: the upper bound on the domain.

    Returns:
      an average error of the shifted inverse algorithm.
    """
    counts = []
    for _ in progress_bar.track(range(100)):
        counts.append(
            shifted_inverse.shifted_inverse_distinct_count(
                data, _EPSILON.value, 0.001, bound, 0.1
            )
        )
    true_count = distinct_count.true_distinct_count(data)
    errors = [abs(count - true_count) for count in counts]
    errors.sort()
    return sum(errors[20:80]) / 60.0


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    with progress.Progress(transient=True) as progress_bar:
        avail_qty = _prepare_dataset(
            pd.read_csv(_AVAIL_QTY.value),
            user_column="suppkey",
            value_column="availqty",
        )

        bounds = list(range(1, 10000))

        order_date = _prepare_dataset(
            pd.read_csv(_ORDER_DATE.value),
            user_column="custkey",
            value_column="orderdate",
        )
        rich.print("[green]SUCCESS:[/green] the order_date dataset is loaded.")
        rich.print(
            "[green]SUCCESS:[/green] shifted inverse algorithrm error on"
            " order_date is"
            f" {_shifted_inverse_error(order_date, progress_bar, 10000)}."
        )
        rich.print(
            "[green]SUCCESS:[/green] greedy algorithrm error on order_date is"
            f" {_error(order_date, dp.greedy_distinct_count, progress_bar, bounds)}."
        )
        rich.print(
            "[green]SUCCESS:[/green] flow algorithrm error on order_date is"
            f" {_error(order_date, dp.flow_distinct_count, progress_bar, bounds)}."
        )

        receipt_date = _prepare_dataset(
            pd.read_csv(_RECEIPT_DATE.value),
            user_column="custkey",
            value_column="receiptdate",
        )
        rich.print("[green]SUCCESS:[/green] the receipt_date dataset is loaded.")
        rich.print(
            "[green]SUCCESS:[/green] shifted inverse algorithrm error on"
            " receipt_date is"
            f" {_shifted_inverse_error(receipt_date, progress_bar, 10000)}."
        )
        rich.print(
            "[green]SUCCESS:[/green] greedy algorithrm error on receipt_date is"
            f" {_error(receipt_date, dp.greedy_distinct_count, progress_bar, bounds)}."
        )
        rich.print(
            "[green]SUCCESS:[/green] flow algorithrm error on receipt_date is"
            f" {_error(receipt_date, dp.flow_distinct_count, progress_bar, bounds)}."
        )

        bounds = list(range(1, 10000000))

        rich.print("[green]SUCCESS:[/green] the avail_qty dataset is loaded.")
        rich.print(
            "[green]SUCCESS:[/green] shifted inverse algorithrm error on"
            " avail_qty is"
            f" {_shifted_inverse_error(avail_qty, progress_bar, 10000000)}."
        )
        rich.print(
            "[green]SUCCESS:[/green] greedy algorithrm error on avail_qty is"
            f" {_error(avail_qty, dp.greedy_distinct_count, progress_bar, bounds)}."
        )
        rich.print(
            "[green]SUCCESS:[/green] flow algorithrm error on avail_qty is"
            f" {_error(avail_qty, dp.flow_distinct_count, progress_bar, bounds)}."
        )

        extended_price = _prepare_dataset(
            pd.read_csv(_EXTENDED_PRICE.value),
            user_column="suppkey",
            value_column="extendedprice",
        )
        rich.print("[green]SUCCESS:[/green] the extended_price dataset is loaded.")
        rich.print(
            "[green]SUCCESS:[/green] shifted inverse algorithrm error on"
            " extended_price is"
            f" {_shifted_inverse_error(extended_price, progress_bar, 10000000)}."
        )
        rich.print(
            "[green]SUCCESS:[/green] greedy algorithrm error on extended_price is"
            f" {_error(extended_price, dp.greedy_distinct_count, progress_bar, bounds)}."
        )
        rich.print(
            "[green]SUCCESS:[/green] flow algorithrm error on extended_price is"
            f" {_error(extended_price, dp.flow_distinct_count, progress_bar, bounds)}."
        )


if __name__ == "__main__":
    app.run(main)
