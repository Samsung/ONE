/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <luci/ImporterEx.h>
#include <luci_interpreter/Interpreter.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

namespace
{

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

template <typename NodeT> size_t getTensorSize(const NodeT *node)
{
  uint32_t tensor_size = loco::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
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
  if (argc != 5)
  {
    std::cerr
      << "Usage: " << argv[0]
      << " <path/to/circle/model> <num_inputs> <path/to/input/prefix> <path/to/output/file>\n";
    return EXIT_FAILURE;
  }

  const char *filename = argv[1];
  const int32_t num_inputs = atoi(argv[2]);
  const char *input_prefix = argv[3];
  const char *output_file = argv[4];

  // Load model from the file
  luci::ImporterEx importer;
  std::unique_ptr<luci::Module> module = importer.importVerifyModule(filename);
  if (module == nullptr)
  {
    std::cerr << "ERROR: Failed to load '" << filename << "'" << std::endl;
    return EXIT_FAILURE;
  }

  // Create interpreter.
  luci_interpreter::Interpreter interpreter(module.get());

  // Set input.
  // Data for n'th input is read from ${input_prefix}n
  // (ex: Add.circle.input0, Add.circle.input1 ..)
  const auto input_nodes = loco::input_nodes(module->graph());
  if (num_inputs != input_nodes.size())
  {
    // NOTE using num_inputs is actually unnecessary but is kept to preserve interface.
    std::cerr << "ERROR: invalid num_inputs value; should be " << input_nodes.size() << std::endl;
    return EXIT_FAILURE;
  }
  for (int32_t i = 0; i < num_inputs; i++)
  {
    const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[i]);
    std::vector<char> input_data(getTensorSize(input_node));
    readDataFromFile(std::string(input_prefix) + std::to_string(i), input_data.data(),
                     input_data.size());
    interpreter.writeInputTensor(input_node, input_data.data(), input_data.size());
  }

  // Do inference.
  interpreter.interpret();

  // Get output.
  const auto output_nodes = loco::output_nodes(module->graph());
  for (int i = 0; i < module->graph()->outputs()->size(); i++)
  {
    const auto *output_node = loco::must_cast<const luci::CircleOutput *>(output_nodes[i]);
    std::vector<char> output_data(getTensorSize(output_node));
    interpreter.readOutputTensor(output_node, output_data.data(), output_data.size());

    // Output data is written in ${output_file}
    // (ex: Add.circle.output0)
    // Output shape is written in ${output_file}.shape
    // (ex: Add.circle.output0.shape)
    writeDataToFile(std::string(output_file) + std::to_string(i), output_data.data(),
                    output_data.size());
    // In case of Tensor output is Scalar value.
    // The output tensor with rank 0 is treated as a scalar with shape (1)
    if (output_node->rank() == 0)
    {
      writeDataToFile(std::string(output_file) + std::to_string(i) + ".shape", "1", 1);
    }
    else
    {
      auto shape_str = std::to_string(output_node->dim(0).value());
      for (int j = 1; j < output_node->rank(); j++)
      {
        shape_str += ",";
        shape_str += std::to_string(output_node->dim(j).value());
      }
      writeDataToFile(std::string(output_file) + std::to_string(i) + ".shape", shape_str.c_str(),
                      shape_str.size());
    }
  }
  return EXIT_SUCCESS;
}
