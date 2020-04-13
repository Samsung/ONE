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

from PIL import Image
import sys
import struct

if (len(sys.argv) < 3):
    print("Usage: %s <input image file> <output bin file>" % (sys.argv[0]))
    exit(0)

img = Image.open(sys.argv[1])
outfile = sys.argv[2]

print("Image format = ", img.bits, img.size, img.format)

with open(outfile, 'wb') as f:
    f.write(img.tobytes())

print("Done.")
