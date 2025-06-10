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

#ifndef __TFLITE_RUN_TENSOR_DUMPER_H__
#define __TFLITE_RUN_TENSOR_DUMPER_H__

#include <tensorflow/lite/c/c_api.h>

#include <memory>
#include <string>
#include <vector>

namespace tflite
{
class Interpreter;
}

namespace TFLiteRun
{

class TensorDumper
{
private:
  struct Tensor
  {
    int _index;
    std::vector<char> _data;

    Tensor(int index, std::vector<char> &&data) : _index(index), _data(std::move(data)) {}
  };

public:
  TensorDumper();
  void addInputTensors(TfLiteInterpreter &interpreter);
  void addOutputTensors(TfLiteInterpreter &interpreter);
  void dump(const std::string &filename) const;

private:
  std::vector<Tensor> _input_tensors;
  std::vector<Tensor> _output_tensors;
};

} // end of namespace TFLiteRun

#endif // __TFLITE_RUN_TENSOR_DUMPER_H__
