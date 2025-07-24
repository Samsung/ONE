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
#include <random>

namespace
{

using DataBuffer = std::vector<char>;

void generateRandomData(char *data, size_t data_size)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  for (size_t i = 0; i < data_size; ++i)
    data[i] = static_cast<char>(dist(gen));
}

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
  // Parse command line arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [ <model_file> | <model_file> <input_prefix> <output_file> ]\n";
    std::cerr << "Options:\n";
    std::cerr << "  run w/o input/output file  : Generate random input data automatically\n";
    std::cerr << "  <input_prefix>: Prefix for input files (e.g. 'input_' for input_0, input_1...)\n";
    std::cerr << "  <output_file> : Output file name\n";
    return EXIT_FAILURE;
  }

  const char* filename = argv[1];
  bool auto_input = false;
  const char* input_prefix = nullptr;
  const char* output_file = nullptr;
  int32_t num_inputs = 1; // Default number of inputs

  // Check for --auto-input flag
  if (argc == 2) {
    auto_input = true;
  } else if (argc >= 4) {
    input_prefix = argv[2];
    output_file = argv[3];
  }

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
  onert_micro::OMInterpreter interpreter;
  onert_micro::OMConfig config;
  interpreter.importModel(model_data.data(), config);

  num_inputs = interpreter.getNumberOfInputs(); // To initialize input buffers

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
      size_t input_size = interpreter.getInputSizeAt(i);
      
      if (auto_input) {
        generateRandomData(input_data, input_size);
      } else {
        readDataFromFile(std::string(input_prefix) + std::to_string(i), 
                        input_data, input_size);
      }
    }

    // Do inference.
    interpreter.run(config);
  }

  // Get output.
  int num_outputs = 1;
  for (int i = 0; i < num_outputs; i++)
  {
    auto data = interpreter.getOutputDataAt(i);
    size_t output_size = interpreter.getOutputSizeAt(i);

    if (output_file != nullptr) {
      writeDataToFile(std::string(output_file) + std::to_string(i),
                     reinterpret_cast<char*>(data), output_size);
    }
    // Otherwise, output remains in interpreter memory
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
