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

#include <luci_interpreter/GraphBuilderRegistry.h>
#include <luci_interpreter/Interpreter.h>

#include <luci/IR/DataTypeHelper.h>
#include <luci/Importer.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <random>

namespace
{

uint32_t tensor_size_of(const luci::CircleNode *node)
{
  uint32_t tensor_size = luci::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

std::vector<uint8_t> random_data_for(const luci::CircleInput *node)
{
  // allocate data buffer
  std::vector<uint8_t> inputs_data(tensor_size_of(node));
  auto *buffer = inputs_data.data();

  // define size of buffer in elements
  const auto dtype = node->dtype();
  assert(inputs_data.size() % luci::size(dtype) == 0); // FIX ME UNLESS
  const auto element_count = inputs_data.size() / luci::size(dtype);

  // random generator engine
  std::random_device device;
  std::mt19937 engine{device()};

  // fill buffer with random data
  switch (node->dtype())
  {
    case loco::DataType::FLOAT32:
    {
      auto element_buffer = reinterpret_cast<float *>(buffer);

      std::uniform_real_distribution<float> distrib(-3, 3);
      const auto generator = [&distrib, &engine]() { return distrib(engine); };
      std::generate(element_buffer, element_buffer + element_count, generator);

      break;
    }
    case loco::DataType::U8:
    {
      auto element_buffer = buffer;

      std::uniform_int_distribution<uint8_t> distrib(100, 200);
      const auto generator = [&distrib, &engine]() { return distrib(engine); };
      std::generate(element_buffer, element_buffer + element_count, generator);

      break;
    }
    case loco::DataType::S16:
    {
      auto element_buffer = reinterpret_cast<int16_t *>(buffer);

      std::uniform_int_distribution<int16_t> distrib(0, 100);
      const auto generator = [&distrib, &engine]() { return distrib(engine); };
      std::generate(element_buffer, element_buffer + element_count, generator);

      break;
    }
    case loco::DataType::S32:
    {
      auto element_buffer = reinterpret_cast<int32_t *>(buffer);

      std::uniform_int_distribution<int32_t> distrib(0, 100);
      const auto generator = [&distrib, &engine]() { return distrib(engine); };
      std::generate(element_buffer, element_buffer + element_count, generator);

      break;
    }
    case loco::DataType::BOOL:
    {
      // num of bool data type is equivalent to uint8_t num in [0, 1] range
      auto element_buffer = buffer;

      std::uniform_int_distribution<uint8_t> distrib(0, 1);
      const auto generator = [&distrib, &engine]() { return distrib(engine); };
      std::generate(element_buffer, element_buffer + element_count, generator);

      break;
    }
    default:
      // TODO Support other dtypes
      throw std::runtime_error("Unsupported data type, yet!");
  }

  return inputs_data;
}

} // namespace

int entry(int argc, char **argv)
{
  // check arguments
  if (argc != 3 || std::string(argv[1]) != "--model")
  {
    std::cerr << "Usage: " << argv[0] << " --model <path/to/model>" << std::endl;
    return EXIT_FAILURE;
  }

  // open file with model
  const auto model_file = std::string(argv[2]);
  std::ifstream fs(model_file, std::ifstream::binary);
  if (fs.fail())
  {
    std::cerr << "Cannot open model file \"" << model_file << "\"." << std::endl;
    return EXIT_FAILURE;
  }

  // create constant circle model
  const std::vector<char> model_buffer((std::istreambuf_iterator<char>(fs)),
                                       std::istreambuf_iterator<char>());
  const auto circle_model = circle::GetModel(model_buffer.data());

  // create random model's inputs
  std::vector<std::vector<uint8_t>> inputs_data;
  {
    // model inputs
    auto model = luci::Importer(nullptr).importModule(circle_model);
    const auto inputs = loco::input_nodes(model->graph());

    // create random data for each input
    for (const auto *input : inputs)
    {
      const auto input_node = loco::must_cast<const luci::CircleInput *>(input);
      inputs_data.emplace_back(random_data_for(input_node));
    }
  }

  // interpret given module
  const auto interpret_module_and_compute_output =
    [&](const std::unique_ptr<luci::Module> &module) {
      // create interpreter
      luci_interpreter::Interpreter interpreter(module.get());

      // model's input and output nodes
      const auto input_nodes = loco::input_nodes(module->graph());
      const auto output_nodes = loco::output_nodes(module->graph());

      // set inputs
      for (uint32_t i = 0; i < input_nodes.size(); ++i)
      {
        const auto input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[i]);
        const auto &data = inputs_data.at(i);
        interpreter.writeInputTensor(input_node, data.data(), data.size());
      }

      // do inference
      interpreter.interpret();

      // read outputs
      std::vector<std::vector<uint8_t>> outputs_data;
      for (const auto *node : output_nodes)
      {
        const auto output_node = loco::must_cast<const luci::CircleOutput *>(node);

        // allocate output buffer
        outputs_data.emplace_back(tensor_size_of(output_node));

        auto &data = outputs_data.back();
        interpreter.readOutputTensor(output_node, data.data(), data.size());
      }

      return outputs_data;
    };

  // import with copying, execute and save
  std::vector<std::vector<uint8_t>> outputs_data_1;
  {
    const auto default_source = &luci::GraphBuilderRegistry::get();
    const auto module = luci::Importer(default_source).importModule(circle_model);
    if (not module)
    {
      std::cerr << "Fail to import model with constant copying." << std::endl;
      return EXIT_FAILURE;
    }

    outputs_data_1 = interpret_module_and_compute_output(module);
  }

  // import without copying, execute and save
  std::vector<std::vector<uint8_t>> outputs_data_2;
  {
    const auto optimized_source = luci_interpreter::source_without_constant_copying();
    const auto module = luci::Importer(optimized_source.get()).importModule(circle_model);
    if (not module)
    {
      std::cerr << "Fail to import model without constant copying." << std::endl;
      return EXIT_FAILURE;
    }

    outputs_data_2 = interpret_module_and_compute_output(module);
  }

  // check all tensors are equal
  assert(outputs_data_1.size() == outputs_data_2.size());
  for (uint32_t n = 0; n < outputs_data_1.size(); ++n)
  {
    const auto &output_1 = outputs_data_1.at(n);
    const auto &output_2 = outputs_data_2.at(n);
    assert(output_1.size() == output_2.size());

    for (uint32_t o = 0; o < output_1.size(); ++o)
    {
      if (output_1[o] != output_2[o])
      {
        std::cerr << "Values mismatch in model's output number " << n << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  std::cout << "[TEST PASSED]" << std::endl;
  return EXIT_SUCCESS;
}
