/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <luci_interpreter/Interpreter.h>

#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>

namespace
{

using DataBuffer = std::vector<char>;

void readDataFromFile(const std::string &filename, char *data, size_t data_size)
{
  std::ifstream fs(filename, std::ifstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");
  if (fs.read(data, data_size).fail())
    throw std::runtime_error("Failed to read data from file \"" + filename + "\".\n");
}

void writeDataToFile(const std::string &filename, const char *data, size_t data_size)
{
  std::ofstream fs(filename, std::ofstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");
  if (fs.write(data, data_size).fail())
  {
    throw std::runtime_error("Failed to write data to file \"" + filename + "\".\n");
  }
}

} // namespace

/*
 * @brief EvalDriver main
 *
 *        Driver for testing luci-inerpreter
 *
 */
int entry(int argc, char **argv)
{
  if (argc != 3)
  {
    std::cerr
      << "Usage: " << argv[0]
      << " <path/to/circle/model> <num_inputs> <path/to/input/prefix> <path/to/output/file>\n";
    return EXIT_FAILURE;
  }

  const char *filename = argv[1];
  const int32_t num_inputs = atoi(argv[2]);
//  const char *input_prefix = argv[3];
//  const char *output_file = argv[4];

  std::ifstream file(filename, std::ios::binary | std::ios::in);
  if (!file.good())
  {
    std::string errmsg = "Failed to open file";
    throw std::runtime_error(errmsg.c_str());
  }

  file.seekg(0, std::ios::end);
  auto fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // reserve capacity
  DataBuffer model_data(fileSize);

  // read the data
  file.read(model_data.data(), fileSize);
  if (file.fail())
  {
    std::string errmsg = "Failed to read file";
    throw std::runtime_error(errmsg.c_str());
  }

  // Create interpreter.
  luci_interpreter::Interpreter interpreter(model_data.data(), true);

  // Set input.
  // Data for n'th input is read from ${input_prefix}n
  // (ex: Add.circle.input0, Add.circle.input1 ..)
  int num_inference = 1;
  for (int j = 0; j < num_inference; ++j)
  {
    for (int32_t i = 0; i < num_inputs; i++)
    {
      auto input_data = reinterpret_cast<char *>(interpreter.allocateInputTensor(i));
      *(float*) input_data = 3.1415;
//      readDataFromFile(std::string(input_prefix) + std::to_string(i), input_data,
//                       interpreter.getInputDataSizeByIndex(i));
    }

    // Do inference.
    interpreter.interpret();
  }

  // Get output.
  int num_outputs = 1;
  for (int i = 0; i < num_outputs; i++)
  {
    auto data = interpreter.readOutputTensor(i);
    std::cout << *(float*) data << "\n";
    // Output data is written in ${output_file}
    // (ex: Add.circle.output0)
//    writeDataToFile(std::string(output_file) + std::to_string(i), reinterpret_cast<char *>(data),
//                    interpreter.getOutputDataSizeByIndex(i));
  }
  return EXIT_SUCCESS;
}

int entry(int argc, char **argv);

#ifdef NDEBUG
int main(int argc, char **argv)
{
  try
  {
    return entry(argc, argv);
  }
  catch (const std::exception &e)
  {
    std::cerr << "ERROR: " << e.what() << std::endl;
  }

  return 255;
}
#else  // NDEBUG
int main(int argc, char **argv)
{
  // NOTE main does not catch internal exceptions for debug build to make it easy to
  //      check the stacktrace with a debugger
  return entry(argc, argv);
}
#endif // !NDEBUG
