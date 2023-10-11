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
#include <luci_interpreter/Interpreter.h>
#include <iostream>
#include <vector>
#include "speech_recognition_float.circle.h"
#include "test_data.h"

luci_interpreter::Interpreter interpreter(circle_model_raw, true);
std::vector<float> infer()
{
  auto input_data = reinterpret_cast<float *>(interpreter.allocateInputTensor(0));
  for (int i = 0; i < 1960; ++i)
  {
    *(input_data + i) = test_data[i];
  }
  interpreter.interpret();
  auto data = interpreter.readOutputTensor(0);
  std::vector<float> output;
  for (int i = 0; i < 4; ++i)
  {
    output.push_back(*(reinterpret_cast<float *>(data) + i));
  }
  return output;
}
void print_float(float x)
{
  int tmp = x * 1000000 - static_cast<int>(x) * 1000000;
  std::cout << (tmp >= 0 ? "" : "-") << static_cast<int>(x) << ".";
  int zeros_to_add = 0;
  for (int i = 100000; i >= 1; i = i / 10)
  {
    if (tmp / i != 0)
      break;
    zeros_to_add++;
  }
  for (int i = 0; i < zeros_to_add; ++i)
  {
    std::cout << "0";
  }
  std::cout << (tmp >= 0 ? tmp : -tmp) << "\n";
}
int main()
{
  Timer t;
  t.start();
  auto res = infer();
  t.stop();
  std::cout << "Executed in " << t.read_us() << "us\n";

  // Predictions after inference with TFLite on PC:
  std::cout << "CORRECT PREDICTION\n0.960639\n0.006082\n0.005203\n0.028074\n";

  std::cout << "MODEL PREDICTION\n";
  for (int i = 0; i < 4; ++i)
    print_float(res[i]);
  while (true)
  {
    ThisThread::sleep_for(100ms);
  }
}
