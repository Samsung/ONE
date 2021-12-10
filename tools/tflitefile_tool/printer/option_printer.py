#!/usr/bin/python

# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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


# TODO: Extract to a single Printer class like Printer.print(option)
class OptionPrinter(object):
    def __init__(self, verbose, op_name, options):
        self.verbose = verbose
        self.op_name = op_name
        self.options = options

    def PrintInfo(self, tab=""):
        info = self.GetStringInfoWONL(tab)
        if info is not None:
            print(info)

    # without new line
    def GetStringInfoWONL(self, tab=""):
        if self.verbose < 1 or self.options == 0:
            return None

        option_str = self.GetStringOption()
        if option_str is None:
            return None

        results = [option_str]
        results.append("{}Options".format(tab))
        results.append("{}\t{}".format(tab, option_str))
        return "\n".join(results)

    def GetStringOption(self):
        if (self.op_name == "AVERAGE_POOL_2D" or self.op_name == "MAX_POOL_2D"):
            return "{}, {}, {}".format(
                "Filter W:H = {}:{}".format(self.options.FilterWidth(),
                                            self.options.FilterHeight()),
                "Stride W:H = {}:{}".format(self.options.StrideW(),
                                            self.options.StrideH()),
                "Padding = {}".format(self.GetStringPadding()))
        elif (self.op_name == "CONV_2D"):
            return "{}, {}, {}".format(
                "Stride W:H = {}:{}".format(self.options.StrideW(),
                                            self.options.StrideH()),
                "Dilation W:H = {}:{}".format(self.options.DilationWFactor(),
                                              self.options.DilationHFactor()),
                "Padding = {}".format(self.GetStringPadding()))
        elif (self.op_name == "DEPTHWISE_CONV_2D"):
            # yapf: disable
            return "{}, {}, {}, {}".format(
                "Stride W:H = {}:{}".format(self.options.StrideW(),
                                                 self.options.StrideH()),
                "Dilation W:H = {}:{}".format(self.options.DilationWFactor(),
                                              self.options.DilationHFactor()),
                "Padding = {}".format(self.GetStringPadding()),
                "DepthMultiplier = {}".format(self.options.DepthMultiplier()))
            # yapf: enable
        elif (self.op_name == "STRIDED_SLICE"):
            # yapf: disable
            return "{}, {}, {}, {}, {}".format(
                "begin_mask({})".format(self.options.BeginMask()),
                "end_mask({})".format(self.options.EndMask()),
                "ellipsis_mask({})".format(self.options.EllipsisMask()),
                "new_axis_mask({})".format(self.options.NewAxisMask()),
                "shrink_axis_mask({})".format(self.options.ShrinkAxisMask()))
            # yapf: enable
        else:
            return None

    def GetStringPadding(self):
        if self.options.Padding() == 0:
            return "SAME"
        elif self.options.Padding() == 1:
            return "VALID"
        else:
            return "** wrong padding value **"
