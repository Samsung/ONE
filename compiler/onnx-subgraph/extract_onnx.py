
# Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved

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
import extract_onnx_lib
import torch
import onnx
import re
print("python executed")
extract_onnx_lib.split_onnx_ios('subgraphs_ios.txt','net/vit_large_simplify.onnx')

