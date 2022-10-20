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


# Change tensor name into the one compatible with Linux file system
# '/' is replaced with '_'
# Too long name is sliced to 255 characters
def to_filename(tensor_name):
    assert isinstance(tensor_name, str)
    return tensor_name.replace('/', '_')[-255:]


# Check if attr is valid
def valid_attr(args, attr):
    return hasattr(args, attr) and getattr(args, attr)


# Recursively visit items and round floats with ndigits
def pretty_float(item, ndigits=4):
    if isinstance(item, dict):
        return {k: pretty_float(v) for k, v in item.items()}
    if isinstance(item, list):
        return [pretty_float(x) for x in item]
    if isinstance(item, float):
        return round(item, 4)
    return item
