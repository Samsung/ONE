/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __TFLITE_RUN_TENSOR_LOADER_H__
#define __TFLITE_RUN_TENSOR_LOADER_H__

#include "tflite/TensorView.h"

#include <sys/mman.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace tflite
{
class Interpreter;
}

namespace TFLiteRun
{

class TensorLoader
{
public:
  TensorLoader(TfLiteInterpreter &interpreter);
  void loadDumpedTensors(const std::string &filename);
  void loadRawInputTensors(const std::string &filename);
  const nnfw::tflite::TensorView<float> &getOutput(int tensor_idx) const;

private:
  size_t loadInputTensorsFromRawData();
  size_t loadOutputTensorsFromRawData();
  TfLiteInterpreter &_interpreter;
  std::unique_ptr<float[]> _raw_data;
  std::unordered_map<int, nnfw::tflite::TensorView<float>> _input_tensor_map;
  std::unordered_map<int, nnfw::tflite::TensorView<float>> _output_tensor_map;
};

} // end of namespace TFLiteRun

#endif // __TFLITE_RUN_TENSOR_LOADER_H__
