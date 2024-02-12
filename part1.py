import csv
import time
import collections
import re
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import timeit
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.ticker
from matplotlib import _api
import itertools
import numpy as np

import data_object_classes
from util import time_section

#handler = None

# Copied from https://github.com/matplotlib/matplotlib/blob/710fce3df95e22701bd68bf6af2c8adbc9d67a79/lib/matplotlib/axes/_axes.py#L2555
# Need to modify this so that rotated bar labels start printing at the top of the bar

def bar_label_modified(self, container, labels=None, *, fmt="%g", label_type="edge",
              padding=0, **kwargs):
    """
    Label a bar plot.
    Adds labels to bars in the given `.BarContainer`.
    You may need to adjust the axis limits to fit the labels.
    Parameters
    ----------
    container : `.BarContainer`
        Container with all the bars and optionally errorbars, likely
        returned from `.bar` or `.barh`.
    labels : array-like, optional
        A list of label texts, that should be displayed. If not given, the
        label texts will be the data values formatted with *fmt*.
    fmt : str, default: '%g'
        A format string for the label.
    label_type : {'edge', 'center'}, default: 'edge'
        The label type. Possible values:
        - 'edge': label placed at the end-point of the bar segment, and the
          value displayed will be the position of that end-point.
        - 'center': label placed in the center of the bar segment, and the
          value displayed will be the length of that segment.
          (useful for stacked bars, i.e.,
          :doc:`/gallery/lines_bars_and_markers/bar_label_demo`)
    padding : float, default: 0
        Distance of label from the end of the bar, in points.
    **kwargs
        Any remaining keyword arguments are passed through to
        `.Axes.annotate`.
    Returns
    -------
    list of `.Text`
        A list of `.Text` instances for the labels.
    """

    # want to know whether to put label on positive or negative direction
    # cannot use np.sign here because it will return 0 if x == 0
    def sign(x):
        return 1 if x >= 0 else -1

    _api.check_in_list(['edge', 'center'], label_type=label_type)

    bars = container.patches
    errorbar = container.errorbar
    datavalues = container.datavalues
    orientation = container.orientation

    if errorbar:
        # check "ErrorbarContainer" for the definition of these elements
        lines = errorbar.lines  # attribute of "ErrorbarContainer" (tuple)
        barlinecols = lines[2]  # 0: data_line, 1: caplines, 2: barlinecols
        barlinecol = barlinecols[0]  # the "LineCollection" of error bars
        errs = barlinecol.get_segments()
    else:
        errs = []

    if labels is None:
        labels = []

    annotations = []

    for bar, err, dat, lbl in itertools.zip_longest(
        bars, errs, datavalues, labels
    ):
        (x0, y0), (x1, y1) = bar.get_bbox().get_points()
        xc, yc = (x0 + x1) / 2, (y0 + y1) / 2

        if orientation == "vertical":
            extrema = max(y0, y1) if dat >= 0 else min(y0, y1)
            length = abs(y0 - y1)
        elif orientation == "horizontal":
            extrema = max(x0, x1) if dat >= 0 else min(x0, x1)
            length = abs(x0 - x1)

        if err is None:
            endpt = extrema
        elif orientation == "vertical":
            endpt = err[:, 1].max() if dat >= 0 else err[:, 1].min()
        elif orientation == "horizontal":
            endpt = err[:, 0].max() if dat >= 0 else err[:, 0].min()

        if label_type == "center":
            value = sign(dat) * length
        elif label_type == "edge":
            value = extrema

        if label_type == "center":
            xy = xc, yc
        elif label_type == "edge" and orientation == "vertical":
            xy = xc, endpt
        elif label_type == "edge" and orientation == "horizontal":
            xy = endpt, yc

        if orientation == "vertical":
            xytext = 0, sign(dat) * padding
        else:
            xytext = sign(dat) * padding, 0

        if label_type == "center":
            ha, va = "center", "center"
        elif label_type == "edge":
            if orientation == "vertical":
                ha = 'left' # CHANGED THIS LINE
                va = 'top' if dat < 0 else 'bottom'  # also handles NaN
            elif orientation == "horizontal":
                ha = 'right' if dat < 0 else 'left'  # also handles NaN
                va = 'center'

        if np.isnan(dat):
            lbl = ''

        annotation = self.annotate(fmt % value if lbl is None else lbl,
                                   xy, xytext, textcoords="offset points",
                                   ha=ha, va=va, **kwargs)
        annotations.append(annotation)

    return annotations

def int_or_percent_float(x):
    if x == "-":
        return 0
    elif x == "<0.1%":
        return 0.01
    elif x.endswith("%"):
        return float(x[:-1])
    else:
        return int(x)

class AttributeFrequency:
    __slots__ = ("frequency", "objects_with_missing_attribute", "attribute_name")

    def __init__(self, frequency, objects_with_missing_attribute, attribute_name):
        self.frequency = frequency
        self.objects_with_missing_attribute = objects_with_missing_attribute
        self.attribute_name = attribute_name

    @classmethod
    def from_data_objects(cls, data_objects, attribute_name):
        frequency = collections.Counter()
        objects_with_missing_attribute = []

        attribute_value_list_or_value = getattr(data_objects[0], attribute_name)
        if isinstance(attribute_value_list_or_value, list):
            for data_object in data_objects:
                for attribute_value in getattr(data_object, attribute_name):
                    if attribute_value == "":
                        objects_with_missing_attribute.append(data_object)
                    else:
                        frequency[attribute_value] += 1

        else:
            for data_object in data_objects:
                attribute_value = getattr(data_object, attribute_name)
                if attribute_value == "":
                    objects_with_missing_attribute.append(data_object)
                else:
                    frequency[attribute_value] += 1

        return cls(frequency, objects_with_missing_attribute, attribute_name)

    def gen_frequency_info(self):
        output = ""
        output += "".join(f"{attribute}: {count}\n" for attribute, count in self.frequency.most_common(10))
        output += f"total unique {self.attribute_name}: {len(self.frequency)}\n"

        return output

    def create_date_histogram_scatter(self, output_filename, **kwargs):
        sorted_frequency_as_list = sorted(((datetime.strptime(k, "%Y-%m-%d"), v) for k, v in self.frequency.items()), key=lambda x: x[0])
        sorted_frequency = {k: v for k, v in sorted_frequency_as_list}
        self.create_date_histogram_common(output_filename, sorted_frequency, sorted_frequency_as_list, **kwargs)

    def create_date_histogram_scatter_remove_bad_dates(self, output_filename, **kwargs):
        sorted_frequency_as_list = sorted(((datetime.strptime(k, "%Y-%m-%d"), v) for k, v in self.frequency.items() if k != "2000-01-01" and int(k[:4]) >= 2000), key=lambda x: x[0])
        sorted_frequency = {k: v for k, v in sorted_frequency_as_list}
        self.create_date_histogram_common(output_filename, sorted_frequency, sorted_frequency_as_list, **kwargs)

    def create_date_histogram_scatter_second_weekend_drop(self, output_filename, **kwargs):
        sorted_frequency_as_list = sorted(((datetime.strptime(k, "%b %d, %Y"), v) for k, v in self.frequency.items() if k != "-"), key=lambda x: x[0])
        sorted_frequency = {k: v for k, v in sorted_frequency_as_list}
        self.create_date_histogram_common(output_filename, sorted_frequency, sorted_frequency_as_list, **kwargs)

    def create_date_histogram_common(self, output_filename, sorted_frequency, sorted_frequency_as_list, title=None, xlabel=None, ylabel=None):
        start_date = sorted_frequency_as_list[0][0]
        end_date = sorted_frequency_as_list[-1][0]
        date_diff = end_date - start_date

        fig, ax = plt.subplots(figsize=(11, 8.5))
        markerline, stemlines, baseline = ax.stem(sorted_frequency.keys(), sorted_frequency.values(), markerfmt=" ")
        plt.setp(stemlines, 'linewidth', 1)

        ax.set_xticklabels(sorted_frequency.keys(), rotation=45, ha="right")
        ax.margins(y=0.1)
        ax.set_xlabel(xlabel or self.attribute_name, labelpad=5)
        ax.set_ylabel(ylabel or "Frequency")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        locator = mdates.DayLocator(interval=date_diff.days//5)
        locator.MAXTICKS = 100000

        ax.xaxis.set_major_locator(locator)
        fig.autofmt_xdate()

        ax.set_title(title or f"{self.attribute_name} frequency")
        # Save the plot
        fig.subplots_adjust(bottom=0.2)
        fig.savefig(output_filename, bbox_inches='tight')
        plt.close(fig)

    def create_histogram(self, output_filename, linewidth=1, title=None, xlabel=None, ylabel=None):
        sorted_frequency_as_list = [(0, 0)] + sorted(((int(k), v) for k, v in self.frequency.items()), key=lambda x: x[0])
        sorted_frequency_keys = [k for k, v in sorted_frequency_as_list]
        print(f"sorted_frequency_keys: {sorted_frequency_keys}")
        sorted_frequency_values = [v for k, v in sorted_frequency_as_list]

        fig, ax = plt.subplots(figsize=(11, 8.5))
        #    ax.plot((x, x), (x, y))
        markerline, stemlines, baseline = ax.stem(sorted_frequency_keys, sorted_frequency_values, markerfmt=" ")
        plt.setp(stemlines, 'linewidth', linewidth)

        ax.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(base=10, offset=0))
        ax.margins(y=0.1)
        ax.set_xlabel(xlabel or self.attribute_name, labelpad=5)
        ax.set_ylabel(ylabel or "Frequency")

        ax.set_title(title or f"{self.attribute_name} frequency")
        # Save the plot
        fig.subplots_adjust(bottom=0.2)
        fig.savefig(output_filename, bbox_inches='tight')
        plt.close(fig)

    def create_percent_histogram(self, output_filename, linewidth=1, title=None, xlabel=None, ylabel=None):
        sorted_frequency_as_list = [(0, 0)] + sorted(((int_or_percent_float(k), v) for k, v in self.frequency.items() if k not in ("-", "100%")), key=lambda x: x[0])
        sorted_frequency_keys = [k for k, v in sorted_frequency_as_list]
        sorted_frequency_values = [v for k, v in sorted_frequency_as_list]
        print(f"sorted_frequency_keys: {sorted_frequency_keys}\nsorted_frequency_values: {sorted_frequency_values}")

        fig, ax = plt.subplots(figsize=(11, 8.5))
        #    ax.plot((x, x), (x, y))
        markerline, stemlines, baseline = ax.stem(sorted_frequency_keys, sorted_frequency_values, markerfmt=" ")
        plt.setp(stemlines, 'linewidth', linewidth)

        ax.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(base=10, offset=0))
        ax.margins(y=0.1)
        ax.set_xlabel(xlabel or self.attribute_name, labelpad=5)
        ax.set_ylabel(ylabel or "Frequency")

        ax.set_title(title or f"{self.attribute_name} frequency (excluding 0% and 100%)")
        # Save the plot
        fig.subplots_adjust(bottom=0.2)
        fig.savefig(output_filename, bbox_inches='tight')
        plt.close(fig)
        

    def create_percent_frequency_graph(self, output_filename, label_rotation=None, xlabelpad=None, plot_bottom=None, bar_label_rotation=None, graph_title=None, xlabel=None, ylabel=None, title=None):
        with time_section(f"create_percent_frequency_graph {self.attribute_name}"):
            total_items = sum(self.frequency.values())

            frequency_as_list_of_tuples = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
            #if max_plot_items is not None:
            #    frequency_as_list_of_tuples = frequency_as_list_of_tuples[:max_plot_items]
            frequency_keys = [k for k, v in frequency_as_list_of_tuples]
            frequency_values_pretty = [f"{v*100/total_items:.2f}% ({v})" for k, v in frequency_as_list_of_tuples]
            frequency_percent_values = [v/total_items for k, v in frequency_as_list_of_tuples]

            fig, ax = plt.subplots(figsize=(11, 8.5))

            bar_container = ax.bar(frequency_keys, frequency_percent_values)
            if label_rotation not in ("vertical", None):
                ax.set_xticklabels(frequency_keys, rotation=label_rotation, ha="right")
            else:
                ax.set_xticklabels(frequency_keys, rotation=label_rotation)

            # add exact amounts on each bar
            if bar_label_rotation is not None:
                bar_label_modified(ax, bar_container, labels=frequency_values_pretty, label_type="edge", rotation=bar_label_rotation)
            else:
                ax.bar_label(bar_container, labels=frequency_values_pretty, label_type="edge")

            ax.margins(y=0.1)
            # Set labels and title
            ax.set_xlabel(xlabel or self.attribute_name, labelpad=xlabelpad)
            ax.set_ylabel(ylabel or "Frequency")
            ax.set_title(graph_title or title or f"{self.attribute_name} frequency")
            # Save the plot
            fig.subplots_adjust(bottom=plot_bottom)
            fig.savefig(output_filename, bbox_inches='tight')
            plt.close(fig)

    def create_frequency_graph(self, output_filename, label_rotation=None, sort_plot=False, max_plot_items=None, xlabelpad=None, plot_bottom=None, bar_label_rotation=None, graph_title=None, xlabel=None, ylabel=None, title=None):
        with time_section(f"create_frequency_graph {self.attribute_name}"):
            # Create a graph using matplotlib
            # 10x10 inches
            
            # bar graph
            if sort_plot:
                frequency_as_list_of_tuples = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
                if max_plot_items is not None:
                    frequency_as_list_of_tuples = frequency_as_list_of_tuples[:max_plot_items]
                frequency = {k: v for k, v in frequency_as_list_of_tuples}
            else:
                if max_plot_items is not None:
                    frequency_as_tuple_of_tuples = tuple(self.frequency.items())[:max_plot_items]
                    frequency = {k: v for k, v in frequency_as_tuple_of_tuples}
                else:
                    frequency = self.frequency

            fig, ax = plt.subplots(figsize=(11, 8.5))

            bar_container = ax.bar(frequency.keys(), frequency.values())
            if label_rotation not in ("vertical", None):
                ax.set_xticklabels(frequency.keys(), rotation=label_rotation, ha="right")
            else:
                ax.set_xticklabels(frequency.keys(), rotation=label_rotation)

            # add exact amounts on each bar
            if bar_label_rotation is not None:
                bar_label_modified(ax, bar_container, labels=frequency.values(), label_type="edge", rotation=bar_label_rotation)
            else:
                ax.bar_label(bar_container, labels=frequency.values(), label_type="edge")
                
            ax.margins(y=0.1)
            # Set labels and title
            ax.set_xlabel(xlabel or self.attribute_name, labelpad=xlabelpad)
            if ylabel is None:
                ax.set_ylabel("Frequency")
            else:
                ax.set_ylabel(ylabel)

            if graph_title is None:
                ax.set_title(title or f"{self.attribute_name} frequency")
            else:
                ax.set_title(graph_title)

            # Save the plot
            fig.subplots_adjust(bottom=plot_bottom)
            fig.savefig(output_filename, bbox_inches='tight')
            plt.close(fig)

def int_str_to_0(x):
    if x not in ("-", ""):
        return int(x)
    else:
        return 0

def create_bar_graph_xaxis_attribute2_yaxis_attribute1(data_objects, output_filename, attribute_name, attribute2_name, label_rotation=None, max_plot_items=None, xlabelpad=None, plot_bottom=None, bar_label_rotation=None, title=None, xlabel=None, ylabel=None):
    attribute1_name = attribute_name

    with time_section(f"create_bar_graph_xaxis_attribute2_yaxis_attribute1 {attribute1_name}"):
        attribute1s_by_attribute2s = [(getattr(data_object, attribute2_name), int_str_to_0(getattr(data_object, attribute1_name))) for data_object in data_objects]
        sorted_attribute1s_by_attribute2s = sorted(attribute1s_by_attribute2s, key=lambda x: x[1], reverse=True)
        if max_plot_items is not None:
            sorted_attribute1s_by_attribute2s = sorted_attribute1s_by_attribute2s[:max_plot_items]

        attribute1s = [k for k, v in sorted_attribute1s_by_attribute2s]
        attribute2s = [v for k, v in sorted_attribute1s_by_attribute2s]
        print(f"attribute1s: {attribute1s}")

        fig, ax = plt.subplots(figsize=(11, 8.5))

        bar_container = ax.bar(attribute1s, attribute2s)
        if label_rotation not in ("vertical", None):
            ax.set_xticklabels(attribute1s, rotation=label_rotation, ha="right")
        else:
            ax.set_xticklabels(attribute1s, rotation=label_rotation)

        # add exact amounts on each bar
        if bar_label_rotation is not None:
            bar_label_modified(ax, bar_container, labels=attribute2s, label_type="edge", rotation=bar_label_rotation)        
        else:
            ax.bar_label(bar_container, labels=attribute2s, label_type="edge")
        ax.margins(y=0.1)
        # Set labels and title
        ax.set_xlabel(xlabel or attribute2_name, labelpad=xlabelpad)
        ax.set_ylabel(ylabel or attribute1_name)
        num_plot_items = len(sorted_attribute1s_by_attribute2s)
        if max_plot_items is not None:
            ax.set_title(title or f"Top {max_plot_items} {attribute2_name} by {attribute1_name}")
        else:
            ax.set_title(title or f"{attribute2_name} by {attribute1_name}")

        # Save the plot
        fig.subplots_adjust(bottom=plot_bottom)
        fig.savefig(output_filename, bbox_inches="tight")
        plt.close(fig)

def create_bar_graph_xaxis_attribute2_yaxis_attribute1_order_by_attribute3(data_objects, output_filename, attribute_name, attribute2_name, attribute3_name, label_rotation=None, max_plot_items=None, xlabelpad=None, plot_bottom=None, bar_label_rotation=None, title=None, xlabel=None, ylabel=None):
    attribute1_name = attribute_name

    with time_section(f"create_bar_graph_xaxis_attribute2_yaxis_attribute1 {attribute1_name}"):
        attribute1s_by_attribute2s = [(getattr(data_object, attribute2_name), int_str_to_0(getattr(data_object, attribute1_name)), int(getattr(data_object, attribute3_name))) for data_object in data_objects]
        sorted_attribute1s_by_attribute2s = sorted(attribute1s_by_attribute2s, key=lambda x: x[2])
        if max_plot_items is not None:
            sorted_attribute1s_by_attribute2s = sorted_attribute1s_by_attribute2s[:max_plot_items]

        attribute1s = [k for k, v, _ in sorted_attribute1s_by_attribute2s]
        attribute2s = [v for k, v, _ in sorted_attribute1s_by_attribute2s]
        print(f"attribute1s: {attribute1s}")

        fig, ax = plt.subplots(figsize=(11, 8.5))

        bar_container = ax.bar(attribute1s, attribute2s)
        if label_rotation not in ("vertical", None):
            ax.set_xticklabels(attribute1s, rotation=label_rotation, ha="right")
        else:
            ax.set_xticklabels(attribute1s, rotation=label_rotation)

        # add exact amounts on each bar
        if bar_label_rotation is not None:
            bar_label_modified(ax, bar_container, labels=attribute2s, label_type="edge", rotation=bar_label_rotation)        
        else:
            ax.bar_label(bar_container, labels=attribute2s, label_type="edge")
        ax.margins(y=0.1)
        # Set labels and title
        ax.set_xlabel(xlabel or attribute2_name, labelpad=xlabelpad)
        ax.set_ylabel(ylabel or attribute1_name)
        num_plot_items = len(sorted_attribute1s_by_attribute2s)
        if max_plot_items is not None:
            ax.set_title(title or f"{attribute3_name} of top {max_plot_items} {attribute2_name}")
        else:
            ax.set_title(title or f"{attribute3_name} of {attribute2_name}")

        # Save the plot
        fig.subplots_adjust(bottom=plot_bottom)
        fig.savefig(output_filename, bbox_inches="tight")
        plt.close(fig)

letter_grades = set(("A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"))

def create_review_score_bar_graph(rt_reviews, output_filename, attribute_name, xlabelpad, plot_bottom):
    with time_section(f"create_review_score_bar_graph {attribute_name}"):

        review_score_type = collections.Counter()

        for rt_review in rt_reviews:
            review_score = rt_review.review_score
            if review_score.endswith("/4"):
                review_score_type["Out of 4"] += 1
            elif review_score.endswith("/5"):
                review_score_type["Out of 5"] += 1
            elif review_score.endswith("/10"):
                review_score_type["Out of 10"] += 1
            elif review_score in letter_grades:
                review_score_type["Letter Grade"] += 1
            elif review_score == "":
                review_score_type["No score"] += 1
            else:
                review_score_type["Other"] += 1

        total_items = sum(review_score_type.values())

        review_score_type_as_list_of_tuples = sorted(review_score_type.items(), key=lambda x: x[1], reverse=True)
        review_score_type_keys = [k for k, v in review_score_type_as_list_of_tuples]
        review_score_type_values_pretty = [f"{v*100/total_items:.2f}% ({v})" for k, v in review_score_type_as_list_of_tuples]
        review_score_type_percent_values = [v/total_items for k, v in review_score_type_as_list_of_tuples]

        fig, ax = plt.subplots(figsize=(11, 8.5))

        bar_container = ax.bar(review_score_type_keys, review_score_type_percent_values)
        label_rotation = None
        bar_label_rotation = None
        if label_rotation not in ("vertical", None):
            ax.set_xticklabels(review_score_type_keys, rotation=label_rotation, ha="right")
        else:
            ax.set_xticklabels(review_score_type_keys, rotation=label_rotation)

        # add exact amounts on each bar
        if bar_label_rotation is not None:
            bar_label_modified(ax, bar_container, labels=review_score_type_values_pretty, label_type="edge", rotation=bar_label_rotation)
        else:
            ax.bar_label(bar_container, labels=review_score_type_values_pretty, label_type="edge")

        ax.margins(y=0.1)
        # Set labels and title
        ax.set_xlabel("Review Score Type", labelpad=xlabelpad)
        ax.set_ylabel("# of movies with Review Score Type")
        ax.set_title("Review Score Type distribution")
        # Save the plot
        fig.subplots_adjust(bottom=plot_bottom)
        fig.savefig(output_filename, bbox_inches='tight')
        plt.close(fig)

def create_second_weekend_drop_graph(second_week_drops, output_filename, attribute_name, rt_movies):
    with time_section(f"create_second_weekend_drop_graph {attribute_name}"):
        rt_movies_by_name_year = collections.defaultdict(dict)

        for rt_movie in rt_movies:
            # ambiguous movies which aren't even in second weekend drop
            if rt_movie.movie_title in ("Hamlet", "Noise", "Social Animals", "Summerland"):
                continue

            original_release_date = rt_movie.original_release_date
            streaming_release_date = rt_movie.streaming_release_date

            if original_release_date != "" and streaming_release_date != "":
                original_year = int(original_release_date[:4])
                streaming_year = int(streaming_release_date[:4])
                year = str(min(original_year, streaming_year))
            if original_release_date != "":
                year = original_release_date[:4]
            elif streaming_release_date != "":
                year = streaming_release_date[:4]
            else:
                year = "-"
            
            rt_movie_by_name_year = rt_movies_by_name_year[rt_movie.movie_title]
            if year in rt_movie_by_name_year:
                raise RuntimeError(f"Ambiguous movie/year in RT movies! movie title: {rt_movie.movie_title}")

            rt_movie_by_name_year[year] = rt_movie

        tomatometer_to_second_week_drop = []

        for second_weekend_drop in second_week_drops:
            possible_rt_movies = rt_movies_by_name_year.get(second_weekend_drop.title)
            if possible_rt_movies is not None:
                if len(possible_rt_movies) == 1:
                    rt_movie = next(iter(possible_rt_movies.values()))
                else:
                    if second_weekend_drop.wide_date == "-":
                        year = second_weekend_drop.wide_date
                    else:
                        year = second_weekend_drop.wide_date.split(", ")[1]

                    rt_movie = possible_rt_movies.get(year)
                    if rt_movie is None:
                        print(f"Warning: Movie {second_weekend_drop.title} ({year}) not in RT movies.")
                        continue

                if second_weekend_drop.percent_change == "<0.1%":
                    percent_change = 0.01
                else:
                    percent_change = float(second_weekend_drop.percent_change[:-1])
                tomatometer_to_second_week_drop.append((int_str_to_0(rt_movie.tomatometer_rating), percent_change))
            else:
                print(f"Warning: Movie {second_weekend_drop.title} not in RT movies.")

        tomatometer_to_second_week_drop_keys = [k for k, v in tomatometer_to_second_week_drop]
        tomatometer_to_second_week_drop_values = [v for k, v in tomatometer_to_second_week_drop]

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.scatter(tomatometer_to_second_week_drop_keys, tomatometer_to_second_week_drop_values, marker="x")

        ax.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(base=10, offset=0))
        ax.margins(y=0.1)
        ax.set_xlabel("Tomatometer Rating", labelpad=5)
        ax.set_ylabel("Second Weekend Drop % Change")

        ax.set_title(f"Tomatometer Rating to Second Weekend Drop % Change")
        # Save the plot
        fig.subplots_adjust(bottom=0.2)
        fig.savefig(output_filename, bbox_inches='tight')
        plt.close(fig)

class AttributeFreqGraphInfo:
    __slots__ = ("data_objects", "plot_function", "attribute_name", "create_graph", "order", "plot_kwargs")

    def __init__(self, order, data_objects, plot_function, attribute_name, create_graph=True, **kwargs):
        if not isinstance(order, int):
            raise RuntimeError()
        self.order = order
        self.data_objects = data_objects
        self.plot_function = plot_function
        self.attribute_name = attribute_name
        self.create_graph = create_graph
        self.plot_kwargs = kwargs

class AttributeGraphInfo:
    __slots__ = ("data_objects", "plot_function", "attribute_name", "order", "create_graph", "plot_kwargs")

    def __init__(self, order, data_objects, plot_function, attribute_name, create_graph=True,**kwargs):
        if not isinstance(order, int):
            raise RuntimeError()
        self.order = order
        self.data_objects = data_objects
        self.plot_function = plot_function
        self.attribute_name = attribute_name
        self.create_graph = create_graph
        self.plot_kwargs = kwargs

def main():
    #global handler
    output = ""

    handler = data_object_classes.DataObjectsHandler(
        "rotten_tomatoes_movies.csv",
        "rotten_tomatoes_critic_reviews.csv",
        "worldwide_and_domestic_lifetime_box_data_2022-01-27.csv",
        "second_weekend_drop_2022-01-27.csv"
    )

    rt_movies, rt_reviews, lifetime_grosses, second_week_drops = handler.get_all_data_objects()
    #print(f"rt_movies: {rt_movies}, rt_reviews: {rt_reviews}")

    #runtime_str = "runtime"

    #print(f"getattr time: {timeit.timeit(lambda: sum(str_to_0(getattr(rt_movie, runtime_str)) for rt_movie in rt_movies), number=100)}")
    #rt_movies_dict_list = [rt_movie.as_dict for rt_movie in rt_movies]
    #print(f"dict time: {timeit.timeit(lambda: sum(str_to_0(rt_movie[runtime_str]) for rt_movie in rt_movies_dict_list), number=100)}")

    attributes_to_get_frequency = (
        AttributeFreqGraphInfo(1, rt_movies, AttributeFrequency.create_percent_frequency_graph, "content_rating", graph_title="Content rating distribution of movies", xlabel="Content Rating", ylabel="# of movies with content rating"),
        AttributeFreqGraphInfo(2, rt_movies, AttributeFrequency.create_percent_frequency_graph, "genres", label_rotation=45, xlabelpad=5, plot_bottom=0.2, bar_label_rotation=45, graph_title="Genre distribution of movies", xlabel="Genre", ylabel="# of movies with genre"),
        AttributeFreqGraphInfo(3, rt_movies, AttributeFrequency.create_frequency_graph, "directors", label_rotation=45, sort_plot=True, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, graph_title="Top 30 directors by most movies directed", xlabel="Directors", ylabel="# of movies directed"),
        AttributeFreqGraphInfo(4, rt_movies, AttributeFrequency.create_frequency_graph, "authors", label_rotation=45, sort_plot=True, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, graph_title="Top 30 authors by most movies authored", xlabel="Authors", ylabel="# of movies authored"),
        AttributeFreqGraphInfo(5, rt_movies, AttributeFrequency.create_frequency_graph, "actors", label_rotation=45, sort_plot=True, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, graph_title="Top 30 actors by most movies acted in", xlabel="Actors", ylabel="# of movies acted in"),
        AttributeFreqGraphInfo(6, rt_movies, AttributeFrequency.create_date_histogram_scatter, "original_release_date", title="# of movies released by original release date", xlabel="Original release date", ylabel="# of movies"),
        AttributeFreqGraphInfo(7, rt_movies, AttributeFrequency.create_date_histogram_scatter, "streaming_release_date", title="# of movies released by streaming release date", xlabel="Original streaming date", ylabel="# of movies"),
        AttributeFreqGraphInfo(8, rt_movies, AttributeFrequency.create_histogram, "runtime", title="Runtime distribution of movies", xlabel="Runtime", ylabel="# of movies"),
        AttributeFreqGraphInfo(9, rt_movies, AttributeFrequency.create_frequency_graph, "production_company", label_rotation=45, sort_plot=True, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, graph_title="Top 30 production companies by most movies produced", xlabel="Production company", ylabel="# of movies produced"),
        AttributeFreqGraphInfo(10, rt_movies, AttributeFrequency.create_percent_frequency_graph, "tomatometer_status", title="Tomatometer status distribution", xlabel="Tomatometer status", ylabel="# of movies with status"),
        AttributeFreqGraphInfo(11, rt_movies, AttributeFrequency.create_histogram, "tomatometer_rating", linewidth=2, title="Tomatometer rating distribution", xlabel="Tomatometer rating", ylabel="# of movies with rating"),
        AttributeFreqGraphInfo(13, rt_movies, AttributeFrequency.create_percent_frequency_graph, "audience_status", title="Audience status distribution", xlabel="Audience status", ylabel="# of movies with status"),
        AttributeFreqGraphInfo(14, rt_movies, AttributeFrequency.create_histogram, "audience_rating", linewidth=2, title="Audience rating distribution", xlabel="Audience rating", ylabel="# of movies with rating"),

        AttributeFreqGraphInfo(17, rt_reviews, AttributeFrequency.create_frequency_graph, "critic_name", label_rotation=45, sort_plot=True, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, bar_label_rotation=45, title="Top 30 critics by most movies reviewed", xlabel="Critic", ylabel="# of movies reviewed"),
        AttributeFreqGraphInfo(18, rt_reviews, AttributeFrequency.create_percent_frequency_graph, "top_critic", xlabelpad=5, plot_bottom=0.2, title="Top critic distribution across all reviews", xlabel="Is top critic?", ylabel="Frequency"),
        AttributeFreqGraphInfo(19, rt_reviews, AttributeFrequency.create_frequency_graph, "publisher_name", label_rotation=45, sort_plot=True, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, bar_label_rotation=45, title="Top 30 publishers by # of reviews published", xlabel="Publisher", ylabel="# of reviews published"),
        AttributeFreqGraphInfo(20, rt_reviews, AttributeFrequency.create_percent_frequency_graph, "review_type", xlabelpad=5, plot_bottom=0.2, title="Tomatometer status distribution across all reviews", xlabel="Tomatometer status", ylabel="# of reviews with status"),
        AttributeFreqGraphInfo(22, rt_reviews, AttributeFrequency.create_date_histogram_scatter_remove_bad_dates, "review_date", title="# of reviews by publication date", xlabel="Publication date", ylabel="# of reviews"),
        AttributeFreqGraphInfo(25, lifetime_grosses, AttributeFrequency.create_percent_histogram, "domestic_percent", title="# of movies by domestic % of lifetime worldwide gross", xlabel="Domestic %", ylabel="# of movies"),
        AttributeFreqGraphInfo(27, lifetime_grosses, AttributeFrequency.create_percent_histogram, "foreign_percent", title="# of movies by foreign % of lifetime worldwide gross", xlabel="Foreign %", ylabel="# of movies"),
        AttributeFreqGraphInfo(32, second_week_drops, AttributeFrequency.create_date_histogram_scatter_second_weekend_drop, "wide_date", title="# of movies by wide date release", xlabel="Wide date", ylabel="# of movies"),
    )

    attributes_to_graph_generically = (
        AttributeGraphInfo(12, rt_movies, create_bar_graph_xaxis_attribute2_yaxis_attribute1,
            attribute_name="tomatometer_count", attribute2_name="movie_title", label_rotation=45, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, title="Top 30 movies by tomatometer count (# of critic reviews)", xlabel="Movie", ylabel="Tomatometer count"
        ),
        AttributeGraphInfo(15, rt_movies, create_bar_graph_xaxis_attribute2_yaxis_attribute1,
            attribute_name="audience_count", attribute2_name="movie_title", label_rotation=45, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, bar_label_rotation=45, title="Top 30 movies by audience count (# of audience reviews)", xlabel="Movie", ylabel="Audience count"
        ),
        AttributeGraphInfo(16, rt_movies, create_bar_graph_xaxis_attribute2_yaxis_attribute1,
            attribute_name="tomatometer_top_critics_count", attribute2_name="movie_title", label_rotation=45, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, title="Top 30 movies by Top Critics count (# of Top Critic reviews)", xlabel="Movie", ylabel="Top Critics count"
        ),
        AttributeGraphInfo(21, rt_reviews, create_review_score_bar_graph, "review_score", xlabelpad=5, plot_bottom=0.2),
        AttributeGraphInfo(23, lifetime_grosses, create_bar_graph_xaxis_attribute2_yaxis_attribute1,
            attribute_name="ww_lifetime_gross", attribute2_name="title", label_rotation=45, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, bar_label_rotation=45, title="Top 30 movies by worldwide lifetime gross", xlabel="Movie", ylabel="Worldwide lifetime gross"
        ),
        AttributeGraphInfo(24, lifetime_grosses, create_bar_graph_xaxis_attribute2_yaxis_attribute1,
            attribute_name="domestic_lifetime_gross", attribute2_name="title", label_rotation=45, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, bar_label_rotation=45, title="Top 30 movies by domestic lifetime gross", xlabel="Movie", ylabel="Domestic lifetime gross"
        ),
        AttributeGraphInfo(26, lifetime_grosses, create_bar_graph_xaxis_attribute2_yaxis_attribute1,
            attribute_name="foreign_lifetime_gross", attribute2_name="title", label_rotation=45, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, bar_label_rotation=45, title="Top 30 movies by foreign lifetime gross", xlabel="Movie", ylabel="Foreign lifetime gross"
        ),
        AttributeGraphInfo(28, second_week_drops, create_bar_graph_xaxis_attribute2_yaxis_attribute1,
            attribute_name="opening_weekend_gross", attribute2_name="title", label_rotation=45, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, bar_label_rotation=45, title="Top 30 movies by opening weekend gross", xlabel="Movie", ylabel="Opening weekend gross"
        ),
        AttributeGraphInfo(29, second_week_drops, create_second_weekend_drop_graph,
            attribute_name="percent_change", rt_movies=rt_movies
        ),
        AttributeGraphInfo(30, second_week_drops, create_bar_graph_xaxis_attribute2_yaxis_attribute1,
            attribute_name="second_weekend_gross", attribute2_name="title", label_rotation=45, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, bar_label_rotation=45, title="Top 30 movies by second weekend gross", xlabel="Movie", ylabel="Second weekend gross"
        ),
        AttributeGraphInfo(31, second_week_drops, create_bar_graph_xaxis_attribute2_yaxis_attribute1_order_by_attribute3,
            attribute_name="theatres", attribute2_name="title", attribute3_name="rank", label_rotation=45, max_plot_items=30, xlabelpad=5, plot_bottom=0.2, title="Theatre count of top 30 movies by greatest second weekend drop", xlabel="Movie", ylabel="# of theatres"
        ),
    )

    for i, attribute_frequency_info in enumerate(attributes_to_get_frequency):
        data_objects = attribute_frequency_info.data_objects
        attribute_name = attribute_frequency_info.attribute_name

        with time_section(f"process {attribute_name} frequency"):
            if data_objects:
                frequency = AttributeFrequency.from_data_objects(data_objects, attribute_name)
                output += f"{frequency.gen_frequency_info()}\n"
                if attribute_frequency_info.create_graph:
                    attribute_frequency_info.plot_function(frequency,
                        f"{attribute_frequency_info.order}_{attribute_name}_frequency_bar_graph.pdf",
                        **attribute_frequency_info.plot_kwargs
                        #label_rotation=attribute_frequency_info.label_rotation,
                        #sort_plot=attribute_frequency_info.sort_plot,
                        #max_plot_items=attribute_frequency_info.max_plot_items,
                        #xlabelpad=attribute_frequency_info.xlabelpad,
                        #plot_bottom=attribute_frequency_info.plot_bottom
                    )

    for attribute_graph_info in attributes_to_graph_generically:
        data_objects = attribute_graph_info.data_objects
        attribute_name = attribute_graph_info.attribute_name

        with time_section(f"process {attribute_name} generic graph"):
            if data_objects and attribute_graph_info.create_graph:
                attribute_graph_info.plot_function(
                    data_objects,
                    f"{attribute_graph_info.order}_{attribute_name}_graph.pdf",
                    attribute_name,
                    **attribute_graph_info.plot_kwargs
                )

    #content_rating_frequency = rt_movies.count_attribute_frequency("content_rating")
    #genres_frequency = rt_movies.count_attribute_frequency("genres")
    #director_frequency = rt_movies.count_attribute_frequency("directors")
    #authors_frequency = rt_movies.count_attribute_frequency("authors")
    #actors_frequency = rt_movies.count_attribute_frequency("actors")
    #tomatometer_status_frequency = rt_movies.count_attribute_frequency("tomatometer_status")
    #tomatometer_top_critics_count_frequency = rt_movies.count_attribute_frequency("tomatometer_top_critics_count")
    #tomatometer_fresh_critics_count_frequency = rt_movies.count_attribute_frequency("tomatometer_fresh_critics_count")
    #tomatometer_rotten_critics_count_frequency = rt_movies.count_attribute_frequency("tomatometer_rotten_critics_count")
    #
    #output += f"{content_rating_frequency.gen_frequency_info()}\n"
    #output += f"{genres_frequency.gen_frequency_info()}\n"
    #output += f"{director_frequency.gen_frequency_info()}\n"
    #output += f"{authors_frequency.gen_frequency_info()}\n"
    #output += f"{actors_frequency.gen_frequency_info()}\n"
    #output += f"{tomatometer_status_frequency.gen_frequency_info()}\n"
    #output += f"{tomatometer_top_critics_count_frequency.gen_frequency_info()}\n"
    #output += f"{tomatometer_fresh_critics_count_frequency.gen_frequency_info()}\n"
    #output += f"{tomatometer_rotten_critics_count_frequency.gen_frequency_info()}\n"

    #for rt_movie in rt_movies.movies:
        

    #output += "\n"
    #output += "Movies missing director field:\n"
    #output += "\n".join(movies_with_missing_director) + "\n"

    print(output)

if __name__ == "__main__":
    main()
