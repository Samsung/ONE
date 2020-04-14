# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (c) 2016, 2017 ARM Limited.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import collections
import os.path
import re
import subprocess
import glob


def resolve_includes(target, source):
    # File collection
    FileEntry = collections.namedtuple('FileEntry', 'target_name file_contents')

    # Include pattern
    pattern = re.compile("#include \"(.*)\"")

    # Get file contents
    files = []
    for i in range(len(source)):
        src = source[i]
        dst = target[i]
        f = open(src)
        cts = f.read()
        f.close()
        contents = cts.splitlines()
        entry = FileEntry(target_name=dst, file_contents=contents)
        files.append((os.path.basename(src), entry))

    # Create dictionary of tupled list
    files_dict = dict(files)

    # Check for includes (can only be files in the same folder)
    final_files = []
    for file in files:
        done = False
        tmp_file = file[1].file_contents
        print(file[1].target_name)
        while not done:
            file_count = 0
            updated_file = []
            for line in tmp_file:
                found = pattern.search(line)
                if found:
                    include_file = found.group(1)
                    data = files_dict[include_file].file_contents
                    updated_file.extend(data)
                else:
                    updated_file.append(line)
                    file_count += 1

            # Check if all include are replaced.
            if file_count == len(tmp_file):
                done = True

            # Update temp file
            tmp_file = updated_file

        # Append and prepend string literal identifiers and add expanded file to final list
        tmp_file.insert(0, "R\"(\n")
        tmp_file.append("\n)\"")
        entry = FileEntry(target_name=file[1].target_name, file_contents=tmp_file)
        final_files.append((file[0], entry))

    # Write output files
    for file in final_files:
        with open(file[1].target_name, 'w+') as out_file:
            out_file.write("\n".join(file[1].file_contents))


# Generate embed files
cl_files = glob.glob('src/core/CL/cl_kernels/*.cl')
cl_files += glob.glob('src/core/CL/cl_kernels/*.h')

# DEBUG: print cl files
print("cl_files:")
print(cl_files)

embed_files = [f + "embed" for f in cl_files]
print("embed_files:")
print(embed_files)

resolve_includes(embed_files, cl_files)
