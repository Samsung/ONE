#!/usr/bin/python

# Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

from .tflite_enum_str_maps import EnumStrMaps


def GetAttribute(o, *args):
    import functools
    return functools.reduce(getattr, args, o)


def BuildBuiltinOptionGen():
    bo_gen = {}
    for val_enum in EnumStrMaps.BuiltinOptions:
        val_str = EnumStrMaps.BuiltinOptions[val_enum]
        try:
            # Dynamically import Builtin Option classes
            # 0 (NONE) is the only exception that does not have no corresponding flatbuffer-generated class
            module = __import__("tflite." + val_str)
            bo_gen[val_enum] = GetAttribute(module, val_str, val_str)
        except ImportError as e:
            assert val_enum == 0 and val_str == "NONE"
    return bo_gen


class OptionLoader:
    builtinOptionGen = BuildBuiltinOptionGen()

    @staticmethod
    def GetBuiltinOptions(options_type, options_table):
        if (options_table == None) and (options_type != 0):
            print(
                "Bad flatbuffer file: undefined builtin option table with defined option type"
            )
            exit(1)
        options = OptionLoader.builtinOptionGen[options_type]()
        options.Init(options_table.Bytes, options_table.Pos)
        return options


def GetStringPadding(options):
    if options.Padding() == 0:
        return "SAME"
    elif options.Padding() == 1:
        return "VALID"
    else:
        return "** wrong padding value **"


def GetStringOptions(op_name, options):
    if (op_name == "AVERAGE_POOL_2D" or op_name == "MAX_POOL_2D"):
        return "{}, {}, {}".format(
            "Filter W:H = {}:{}".format(options.FilterWidth(), options.FilterHeight()),
            "Stride W:H = {}:{}".format(options.StrideW(), options.StrideH()),
            "Padding = {}".format(GetStringPadding(options)))
    elif (op_name == "CONV_2D"):
        return "{}, {}, {}".format(
            "Stride W:H = {}:{}".format(options.StrideW(), options.StrideH()),
            "Dilation W:H = {}:{}".format(options.DilationWFactor(),
                                          options.DilationHFactor()),
            "Padding = {}".format(GetStringPadding(options)))
    elif (op_name == "DEPTHWISE_CONV_2D"):
        # yapf: disable
        return "{}, {}, {}, {}".format(
            "Stride W:H = {}:{}".format(options.StrideW(),
                                                options.StrideH()),
            "Dilation W:H = {}:{}".format(options.DilationWFactor(),
                                            options.DilationHFactor()),
            "Padding = {}".format(GetStringPadding(options)),
            "DepthMultiplier = {}".format(options.DepthMultiplier()))
        # yapf: enable
    elif (op_name == "STRIDED_SLICE"):
        # yapf: disable
        return "{}, {}, {}, {}, {}".format(
            "begin_mask({})".format(options.BeginMask()),
            "end_mask({})".format(options.EndMask()),
            "ellipsis_mask({})".format(options.EllipsisMask()),
            "new_axis_mask({})".format(options.NewAxisMask()),
            "shrink_axis_mask({})".format(options.ShrinkAxisMask()))
        # yapf: enable
    else:
        return None
