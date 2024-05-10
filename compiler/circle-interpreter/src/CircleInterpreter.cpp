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

#include <arser/arser.h>
#include <luci/ImporterEx.h>
#include <luci/IR/DataTypeHelper.h>
#include <luci_interpreter/Interpreter.h>
#include <vconone/vconone.h>

#include <cstdlib>
#include <fstream>
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
    throw std::runtime_error("Input tensor size mismatches with \"" + filename + "\".\n");
  if (fs.peek() != EOF)
    throw std::runtime_error("Input tensor size mismatches with \"" + filename + "\".\n");
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
  uint32_t tensor_size = luci::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

void print_version(void)
{
  std::cout << "circle-interpreter version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

} // namespace

/*
 * @brief CircleInterpreter main
 *
 *        Driver to invoke luci-interpreter
 *
 */
int entry(int argc, char **argv)
{
  arser::Arser arser("Interpreter driver for circle models");

  arser::Helper::add_version(arser, print_version);

  arser.add_argument("model_path").help("Circle model filepath");
  arser.add_argument("input_prefix")
    .help("Input data filepath for circle model. "
          "n-th input data is read from ${input_prefix}n, "
          "for example, Add.circle.input0, Add.circle.input1");
  arser.add_argument("output_prefix")
    .help("Output data filepath for circle model. "
          "Output data is written in ${output_file}n, "
          "for example, Add.circle.output0");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cout << err.what() << std::endl;
    std::cout << arser;
    return EXIT_FAILURE;
  }

  const auto filename = arser.get<std::string>("model_path");
  const auto input_prefix = arser.get<std::string>("input_prefix");
  const auto output_prefix = arser.get<std::string>("output_prefix");

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
  for (int32_t i = 0; i < input_nodes.size(); i++)
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

    // Output data is written in ${output_file}n
    // (ex: Add.circle.output0)
    writeDataToFile(std::string(output_prefix) + std::to_string(i), output_data.data(),
                    output_data.size());
  }
  return EXIT_SUCCESS;
}
