/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

void print_help()
{
  std::cerr << "This generates a file that contains result of running TensorFlow Lite interpreter"
            << std::endl
            << "Usage:" << std::endl
            << "\t"
            << "$ gen-tflite-output <tflite_file_path> <input_file_path> <output_path_to_generate>"
            << std::endl;
}

void gen_tflite_output(const std::string tflite_path, const std::string input_path,
                       const std::string output_path)
{
  throw std::runtime_error("Not Yet Implemented");
}

} // namespace

int main(int argc, char **argv)
{
  // TODO We need better args parsing in future
  if (argc != 4)
  {
    print_help();
    return 255;
  }

  gen_tflite_output(argv[1], argv[2], argv[3]);

  return 0;
}
