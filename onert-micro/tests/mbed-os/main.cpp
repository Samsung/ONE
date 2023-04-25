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
#include <iostream>
#include <cstring>
#include "speech_recognition.circle.h"
int main()
{
  luci_interpreter::Interpreter interpreter(circle_model_raw, true);
  int num_inference = 1;
  const int32_t num_inputs = 1;

  for (int j = 0; j < num_inference; ++j)
  {
    for (int32_t i = 0; i < num_inputs; i++)
    {
      auto input_data = reinterpret_cast<char *>(interpreter.allocateInputTensor(i));
      std::memset(input_data, 0, interpreter.getInputDataSizeByIndex(i));
    }

    // Do inference.
    Timer t;
    t.start();
    interpreter.interpret();
    t.stop();
    std::cout << "Executed in " << t.read_us() << "us\n";
  }
  // Get output.
  int num_outputs = 1;
  for (int i = 0; i < num_outputs; i++)
  {
    auto data = interpreter.readOutputTensor(i);

  }
  while (true)
  {
    ThisThread::sleep_for(10ms);
//    std::cout << "Hello world\n";
  }
}
