# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

import os, shutil, PIL.Image, numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', type=str, default='img_files')
parser.add_argument('--output_dir', type=str, default='raw_files')
parser.add_argument('--list_file', type=str, default='datalist.txt')
parser.add_argument('--output_extension', type=str, default='data')

args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
list_file = args.list_file
output_extension = args.output_extension

if os.path.exists(output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir)

for (root, _, files) in os.walk(input_dir):
    datalist = open(list_file, 'w')
    for f in files:
        with PIL.Image.open(root + '/' + f) as image:
            img = np.array(image.resize((299, 299),
                                        PIL.Image.ANTIALIAS)).astype(np.float32)
            img = ((img / 255) - 0.5) * 2.0
            f = f.replace('.jpg', '')
            if output_extension:
                f = f + '.' + output_extension
            output_file = output_dir + '/' + f
            img.tofile(output_file)
            datalist.writelines(os.path.abspath(output_file) + '\n')
    datalist.close()
