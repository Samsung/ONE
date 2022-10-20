# Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Class to save colorscheme
class Palette:
    # Child class must implement __init__ to fill the below members
    def __init__(self):
        # Element of self._slots has [lower bound, upper bound] of qerrors to decide a color
        self._slots = []
        # Element of self._colors has rgb values in string format
        self._colors = []
        raise NotImplementedError('Child class must implement __init__')

    # Return color scheme as a list of objects
    # Each object has the following attributes
    # b: begin qerror
    # e: end qerror
    # c: color (in RGB string)
    def colorscheme(self):
        cs = []
        for slot, color in zip(self._slots, self._colors):
            cs.append({"b": slot[0], "e": slot[1], "c": color})
        return cs


# Ranges of slots are defined by qerror_min/qerror_max
# Each slot has a uniform range
# For example, if qerror_min = 0.0, qerror_max = 1.0, number of colors = 10
# Ranges of slots will be as follows.
# [0.0, 0.1], [0.1, 0.2], [0.2, 0.3] ... [0.8, 0.9], [0.9, 1.0]
class UniformPalette(Palette):
    def __init__(self, qerror_min, qerror_max, colors):
        self._colors = colors
        self._slots = []
        qerror_range = qerror_max - qerror_min
        num_colors = len(self._colors)
        for i in range(num_colors):
            lower_bound = qerror_min + i * (qerror_range / num_colors)
            upper_bound = qerror_min + (i + 1) * (qerror_range / num_colors)

            self._slots.append([lower_bound, upper_bound])

        # Invariant
        assert len(self._slots) == num_colors


# Palette for ylorrd9 colorscheme
class YLORRD9Palette(UniformPalette):
    def __init__(self, qerror_min, qerror_max):
        if qerror_min >= qerror_max:
            raise RuntimeError('min must be less than max')

        # From https://colorbrewer2.org/#type=sequential&scheme=YlOrRd&n=9
        colors = [
            "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c",
            "#bd0026", "#800026"
        ]
        super().__init__(qerror_min, qerror_max, colors)
