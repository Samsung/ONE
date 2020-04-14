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


class OptionPrinter(object):
    def __init__(self, verbose, op_name, options):
        self.verbose = verbose
        self.op_name = op_name
        self.options = options

    def GetPadding(self):
        if self.options.Padding() == 0:
            return "SAME"
        elif self.options.Padding() == 1:
            return "VALID"
        else:
            return "** wrong padding value **"

    def PrintInfo(self, tab=""):
        if (self.verbose < 1):
            pass
        if (self.options == 0):
            return

        option_str = self.GetOptionString()
        if option_str:
            print("{}Options".format(tab))
            print("{}\t{}".format(tab, option_str))

    def GetOptionString(self):
        if (self.op_name == "AVERAGE_POOL_2D" or self.op_name == "MAX_POOL_2D"):
            return "{}, {}, {}".format(
                "Filter W:H = {}:{}".format(self.options.FilterWidth(),
                                            self.options.FilterHeight()),
                "Stride W:H = {}:{}".format(self.options.StrideW(),
                                            self.options.StrideH()),
                "Padding = {}".format(self.GetPadding()))
        elif (self.op_name == "CONV_2D"):
            return "{}, {}, {}".format(
                "Stride W:H = {}:{}".format(self.options.StrideW(),
                                            self.options.StrideH()),
                "Dilation W:H = {}:{}".format(self.options.DilationWFactor(),
                                              self.options.DilationHFactor()),
                "Padding = {}".format(self.GetPadding()))
        elif (self.op_name == "DEPTHWISE_CONV_2D"):
            # yapf: disable
            return "{}, {}, {}, {}".format(
                "Stride W:H = {}:{}".format(self.options.StrideW(),
                                                 self.options.StrideH()),
                "Dilation W:H = {}:{}".format(self.options.DilationWFactor(),
                                              self.options.DilationHFactor()),
                "Padding = {}".format(self.GetPadding()),
                "DepthMultiplier = {}".format(self.options.DepthMultiplier()))
            # yapf: enable
