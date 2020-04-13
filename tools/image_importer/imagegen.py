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
import numpy as np

image_size = {
    "H": 10,
    "W": 10,
    "C": 3  # C is fixed as 3 for R,G,B channels
}

rgb = np.zeros([image_size['H'], image_size['W'], image_size["C"]], dtype=np.uint8)
for y in range(image_size["H"]):
    for x in range(image_size["W"]):
        for c in range(image_size["C"]):
            rgb[y][x][c] = 255  #value range = [0~255]

im = Image.fromarray(rgb)
im.save("image.ppm")

# image can be saved as .jpg or .png
# im.save("image.jpg")
# im.save("image.png")

with open("image.bin", "wb") as f:
    f.write(im.tobytes())
