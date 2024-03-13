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

#include "OMInterpreter.h"
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
 *        Driver for testing onert-micro
 *
 */
int entry(int argc, char **argv)
{
  if (argc != 5 and argc != 6)
  {
    std::cerr
      << "Usage: " << argv[0]
      << " optional<<path/to/wof/file>> <path/to/circle/model> <num_inputs> <path/to/input/prefix> <path/to/output/file>\n";
    return EXIT_FAILURE;
  }

  const char *filename = nullptr;
  int32_t num_inputs = 0;
  const char *input_prefix = nullptr;
  const char *output_file = nullptr;
  const char *wof_file_path = nullptr;

  if (argc == 5)
  {
    filename = argv[1];
    num_inputs = atoi(argv[2]);
    input_prefix = argv[3];
    output_file = argv[4];
  } else if (argc == 6)
  {
    wof_file_path = argv[1];
    filename = argv[2];
    num_inputs = atoi(argv[3]);
    input_prefix = argv[4];
    output_file = argv[5];
  } else
  {
    std::string errmsg = "Wrong driver arguments";
    throw std::runtime_error(errmsg.c_str());
  }

  std::ifstream file(filename, std::ios::binary | std::ios::in);
  if (!file.good())
  {
    std::string errmsg = "Failed to open file";
    throw std::runtime_error(errmsg.c_str());
  }

  // For WOF file
  DataBuffer wof_data;

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
  onert_micro::OMInterpreter interpreter;
  onert_micro::OMConfig config;

  // Read wof_data if path is provided
  if (wof_file_path != nullptr)
  {
    std::ifstream wof_file(wof_file_path, std::ios::binary | std::ios::in);
    if (!wof_file.good())
    {
      std::string errmsg = "Failed to open wof file";
      throw std::runtime_error(errmsg.c_str());
    }

    wof_file.seekg(0, std::ios::end);
    auto wof_file_size = wof_file.tellg();
    wof_file.seekg(0, std::ios::beg);

    wof_data.resize(wof_file_size);

    // read wof the data
    wof_file.read(wof_data.data(), wof_file_size);
    if (wof_file.fail())
    {
      std::string errmsg = "Failed to read wof file";
      throw std::runtime_error(errmsg.c_str());
    }

    wof_file.close();

    config.wof_ptr = reinterpret_cast<char *>(wof_data.data());
  }

  interpreter.importModel(model_data.data(), config);

  num_inputs = interpreter.getNumberOfInputs();
  // Set input.
  // Data for n'th input is read from ${input_prefix}n
  // (ex: Add.circle.input0, Add.circle.input1 ..)
  int num_inference = 1;
  for (int j = 0; j < num_inference; ++j)
  {
    interpreter.reset();
    interpreter.allocateInputs();
    for (int32_t i = 0; i < num_inputs; i++)
    {
      auto input_data = reinterpret_cast<char *>(interpreter.getInputDataAt(i));
      readDataFromFile(std::string(input_prefix) + std::to_string(i), input_data,
                       interpreter.getInputSizeAt(i) * sizeof(float));
    }

    // Do inference.
    interpreter.run();
  }

  // Get output.
  int num_outputs = interpreter.getNumberOfOutputs();
  for (int i = 0; i < num_outputs; i++)
  {
    auto data = interpreter.getOutputDataAt(i);

    // Output data is written in ${output_file}
    // (ex: Add.circle.output0)
    writeDataToFile(std::string(output_file) + std::to_string(i), reinterpret_cast<char *>(data),
                    interpreter.getOutputSizeAt(i) * sizeof(float));
  }
  interpreter.reset();
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
