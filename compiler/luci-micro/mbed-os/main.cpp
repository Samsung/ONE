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

#include <luci_interpreter/Interpreter.h>
#include <luci/Importer.h>
#include <luci/IR/Module.h>
#include <loco/IR/DataTypeTraits.h>
#include <circlemodel.h>
#include <iostream>
// Maximum number of element the application buffer can contain
#define MAXIMUM_BUFFER_SIZE 32

// Create a DigitalOutput object to toggle an LED whenever data is received.
static DigitalOut led(LED1);

// Create a BufferedSerial object with a default baud rate.
static BufferedSerial serial_port(USBTX, USBRX);
constexpr auto BLINKING_RATE = 100;
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
    default:
      assert(false);
  }
}
int main()
{
//  std::vector<char> *buf = new std::vector<char>(circle_model_raw, circle_model_raw + sizeof(circle_model_raw) / sizeof(circle_model_raw[0]));
//  std::vector<char> &model_data = *buf;
  // Verify flatbuffers
  flatbuffers::Verifier verifier{reinterpret_cast<const uint8_t *>(circle_model_raw), sizeof(circle_model_raw) / sizeof(circle_model_raw[0])};
  std::cout << "circle::VerifyModelBuffer\n";
  if (!circle::VerifyModelBuffer(verifier))
  {
    std::cout << "ERROR: Failed to verify circle\n";
  }
  std::cout << "OK\n";
  std::cout << "luci::Importer().importModule\n";
  ThisThread::sleep_for(1000);

  auto module = luci::Importer().importModule(circle::GetModel(circle_model_raw));
//  auto interpreter = std::make_unique<luci_interpreter::Interpreter>(module.get());
//  for(int i = 0; i < sizeof(module);++i)
//  {
//    std::cout << std::hex << *(reinterpret_cast<char*>(&module) + i);
//  }
//  auto nodes = module->graph()->nodes();
//  auto nodes_count = nodes->size();
//  std::cout <<  "nodes_count: %d\n";
  // Fill input tensors with some garbage
//  while (true)
//  {
//    Timer t;
//    t.start();
//    for (int i = 0; i < nodes_count; ++i)
//    {
//      auto *node = dynamic_cast<luci::CircleNode *>(nodes->at(i));
//      assert(node);
//      if (node->opcode() == luci::CircleOpcode::CIRCLEINPUT)
//      {
//        auto *input_node = static_cast<luci::CircleInput *>(node);
//        loco::GraphInput *g_input = module->graph()->inputs()->at(input_node->index());
//        const loco::TensorShape *shape = g_input->shape();
//        size_t data_size = 1;
//        for (int d = 0; d < shape->rank(); ++d)
//        {
//          assert(shape->dim(d).known());
//          data_size *= shape->dim(d).value();
//        }
//        data_size *= loco::size(g_input->dtype());
//        std::vector<char> data(data_size);
//        fill_in_tensor(data, g_input->dtype());
//
//        interpreter->writeInputTensor(static_cast<luci::CircleInput *>(node), data.data(),
//                                      data_size);
//      }
//    }

//    interpreter->interpret();
//    t.stop();
//    std::cout << "\rFinished in " << t.read_us();
    ThisThread::sleep_for(10);
//  }
  // Set desired properties (9600-8-N-1).
  serial_port.set_baud(9600);
  serial_port.set_format(
    /* bits */ 8,
    /* parity */ BufferedSerial::None,
    /* stop bit */ 1);

  // Application buffer to receive the data
//  char buf[MAXIMUM_BUFFER_SIZE] = {0};
  // Initialise the digital pin LED1 as an output
#ifdef LED1
  DigitalOut led(LED1);
#else
  bool led;
#endif
  while (true)
  {
//    led = !led;
    ThisThread::sleep_for(BLINKING_RATE);
//    if (uint32_t num = serial_port.read(buf, sizeof(buf)))
//    {
//      // Toggle the LED.
      led = !led;

      // Echo the input back to the terminal.
//      serial_port.write(buf, num);
//    }
  }
}
