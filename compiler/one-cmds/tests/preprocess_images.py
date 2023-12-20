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

input_dir = 'img_files'
raw_output_dir = 'raw_files'
raw_list_file = 'datalist.txt'
numpy_output_dir = 'numpy_files'
numpy_list_file = 'datalist_numpy.txt'

if os.path.exists(raw_output_dir):
    shutil.rmtree(raw_output_dir, ignore_errors=True)
os.makedirs(raw_output_dir)

if os.path.exists(numpy_output_dir):
    shutil.rmtree(numpy_output_dir, ignore_errors=True)
os.makedirs(numpy_output_dir)

for (root, _, files) in os.walk(input_dir):
    raw_datalist = open(raw_list_file, 'w')
    numpy_datalist = open(numpy_list_file, 'w')
    for f in files:
        with PIL.Image.open(root + '/' + f) as image:
            # To handle ANTIALIAS deprecation
            ANTIALIAS = PIL.Image.Resampling.LANCZOS if hasattr(
                PIL.Image, "Resampling") else PIL.Image.ANTIALIAS

            img = np.array(image.resize((299, 299), ANTIALIAS)).astype(np.float32)
            img = ((img / 255) - 0.5) * 2.0
            # To save in numpy format
            output_numpy_file = numpy_output_dir + '/' + f.replace('jpg', 'npy')
            np.save('./' + output_numpy_file, img)
            numpy_datalist.writelines(os.path.abspath(output_numpy_file) + '\n')

            # To save in raw data format
            output_raw_file = raw_output_dir + '/' + f.replace('jpg', 'data')
            img.tofile(output_raw_file)
            raw_datalist.writelines(os.path.abspath(output_raw_file) + '\n')

    raw_datalist.close()
    numpy_datalist.close()
