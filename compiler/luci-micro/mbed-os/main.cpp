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

#include "mbed.h"
#undef ARG_MAX
#define LUCI_LOG 0
#include <luci_interpreter/Interpreter.h>
#include <luci_interpreter/StaticMemoryManager.h>
#include <luci/Importer.h>
#include <luci/IR/Module.h>
#include <loco/IR/DataTypeTraits.h>
#include <circlemodel.h>
#include <cstdlib>
#include <iostream>
#include <luci/Log.h>

void fill_in_tensor(std::vector<char> &data, loco::DataType dtype)
{
  switch (dtype)
  {
    case loco::DataType::FLOAT32:
      for (int i = 0; i < data.size() / sizeof(float); ++i)
      {
        reinterpret_cast<float *>(data.data())[i] = 123.f;
      }
      break;
    case loco::DataType::S8:
      for (int i = 0; i < data.size() / sizeof(int8_t); ++i)
      {
        reinterpret_cast<int8_t *>(data.data())[i] = 123;
      }
      break;
    case loco::DataType::U8:
      for (int i = 0; i < data.size() / sizeof(uint8_t); ++i)
      {
        reinterpret_cast<uint8_t *>(data.data())[i] = 123;
      }
      break;
    default:
      assert(false);
  }
}
int main()
{
  setenv("ONE_HERMES_COLOR", "ON", 1);
  // Verify flatbuffers
  flatbuffers::Verifier verifier{reinterpret_cast<const uint8_t *>(circle_model_raw), sizeof(circle_model_raw) / sizeof(circle_model_raw[0])};

  std::cout << "circle::VerifyModelBuffer\n";
  if (!circle::VerifyModelBuffer(verifier))
  {
    std::cout << "ERROR: Failed to verify circle\n";
  }
  std::cout << "OK\n";
  std::cout << "circle::GetModel(circle_model_raw)\n";
  auto model = circle::GetModel(circle_model_raw);
  std::cout << "luci::Importer().importModule\n";
  auto module = luci::Importer().importModule(model);
  std::cout << "OK\n";
  std::cout << "std::make_unique<luci_interpreter::Interpreter>(module.get())\n";
  auto interpreter = std::make_unique<luci_interpreter::Interpreter>(module.get());

  std::cout << "OK\n";
  auto nodes = module->graph()->nodes();
  auto nodes_count = nodes->size();
  // Fill input tensors with some garbage
  while (true)
  {
    Timer t;
    for (int i = 0; i < nodes_count; ++i)
    {
      auto *node = dynamic_cast<luci::CircleNode *>(nodes->at(i));
      assert(node);
      if (node->opcode() == luci::CircleOpcode::CIRCLEINPUT)
      {
        auto *input_node = static_cast<luci::CircleInput *>(node);
        loco::GraphInput *g_input = module->graph()->inputs()->at(input_node->index());
        const loco::TensorShape *shape = g_input->shape();
        size_t data_size = 1;
        for (int d = 0; d < shape->rank(); ++d)
        {
          assert(shape->dim(d).known());
          data_size *= shape->dim(d).value();
        }
        data_size *= loco::size(g_input->dtype());
        std::vector<char> data(data_size);
        fill_in_tensor(data, g_input->dtype());

        interpreter->writeInputTensor(static_cast<luci::CircleInput *>(node), data.data(),
                                      data_size);
      }
    }
    t.start();

    interpreter->interpret();
    t.stop();
    std::cout << "\rFinished in " << t.read_us();
    ThisThread::sleep_for(10);
  }
}
