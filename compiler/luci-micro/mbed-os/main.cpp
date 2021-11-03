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
int main()
{
  std::vector<char> *buf = new std::vector<char>(circle_model_raw, circle_model_raw + sizeof(circle_model_raw) / sizeof(circle_model_raw[0]));
  std::vector<char> &model_data = *buf;
  // Verify flatbuffers
  flatbuffers::Verifier verifier{static_cast<const uint8_t *>(static_cast<void *>(model_data.data())), model_data.size()};
  std::cout << "circle::VerifyModelBuffer\n";
  if (!circle::VerifyModelBuffer(verifier))
  {
    std::cout << "ERROR: Failed to verify circle\n";
  }
  std::cout << "OK\n";
  std::cout << "luci::Importer().importModule\n";

  auto module = luci::Importer().importModule(circle::GetModel(model_data.data()));
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
