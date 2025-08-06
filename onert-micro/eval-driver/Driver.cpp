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
#include "arser/arser.h"

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
  // Parse command line arguments using arser
  arser::Arser arser;
  arser.add_argument("--model")
    .type(arser::DataType::STR)
    .required(true)
    .help("Path to model.circle file");
  arser.add_argument("--input_prefix")
    .type(arser::DataType::STR)
    .help("Prefix for input files (generates random inputs if not provided)");
  arser.add_argument("--output_prefix").type(arser::DataType::STR).help("Prefix for output files");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    std::cerr << arser;
    return EXIT_FAILURE;
  }

  const auto filename = arser.get<std::string>("--model");
  std::string input_prefix;
  std::string output_prefix;

  if (arser["--input_prefix"])
  {
    input_prefix = arser.get<std::string>("--input_prefix");
  }
  if (arser["--output_prefix"])
  {
    output_prefix = arser.get<std::string>("--output_prefix");
  }
  const bool auto_input = !arser["--input_prefix"];
  int32_t num_inputs = 1; // Default number of inputs

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

      if (auto_input)
      {
        generateRandomData(input_data, input_size);
      }
      else
      {
        readDataFromFile(input_prefix + std::to_string(i), input_data, input_size);
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

    if (arser["--output_prefix"])
    {
      writeDataToFile(output_prefix + std::to_string(i), reinterpret_cast<char *>(data),
                      output_size);
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
