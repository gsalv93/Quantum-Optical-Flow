# Copyright 2022 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import matplotlib.pyplot as plt
from itertools import cycle

# Plot the anneal schedule
def plot_schedule(schedule, title):
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.set_title(title)
    ax.plot(*[[s[i] for s in schedule] for i in [0, 1]])
    plt.xlabel('Time [us]')
    plt.ylabel('Annealing Parameter s')
    plt.show()

# Plot the success fraction of an anneal schedule
all_colors = ["dodgerblue", "orange", "darkorchid"]
all_lines = [[4, 1, 4, 2], [4, 4], [3, 1, 2, 2]]
all_markers = ["o", "s", "^", "D"]

def plot_success_fraction(success_prob, title, group):

    if group == "pause_duration":
        group_label = "pause"
        yrange = [1e-3, 0.65]
    else:
        group_label = "quench slope"
        yrange = [1e-4, 0.5]

    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_title(title)
    plt.xlabel('Start s')
    plt.ylabel('Ground-State Fraction')
    ax.set_yscale('log')
    ax.set_ylim(yrange)

    colors = cycle(all_colors)
    lines = cycle(all_lines)

    for anneal,a_group in success_prob.groupby('anneal_time'):
        col = next(colors)
        line = next(lines)

        markers = cycle(all_markers)
        for pq,p_group in a_group.groupby(group):
            marker = next(markers)

            x = p_group["s_feature"].values
            y = p_group["success_frac"].values
            ax.plot(x, y, label=f"anneal={anneal}, {group_label}={pq}", color=col, dashes=line, linewidth=2)
            ax.scatter(x, y, color=col)

    ax.legend(loc='lower right')
    plt.show()
    return {"figure": fig, "axis": ax}