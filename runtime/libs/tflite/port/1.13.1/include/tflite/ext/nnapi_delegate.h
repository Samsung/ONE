/* Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
   Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// NOTE To minimize diff with upstream tensorflow, disable clang-format
// clang-format off

// NOTE This header is derived from the following file (in TensorFlow v1.13.1)
//        'externals/tensorflow/tensorflow/lite/nnapi_delegate.h'
#ifndef __NNFW_TFLITE_EXT_NNAPI_DELEGATE_H__
#define __NNFW_TFLITE_EXT_NNAPI_DELEGATE_H__

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"

struct ANeuralNetworksModel;
struct ANeuralNetworksMemory;
struct ANeuralNetworksCompilation;

namespace nnfw {
namespace tflite {

class NNAPIAllocation : public ::tflite::MMAPAllocation {
 public:
  NNAPIAllocation(const char* filename, ::tflite::ErrorReporter* error_reporter);
  ~NNAPIAllocation();

  size_t offset(const void* ptr) const {
    auto signed_offset = reinterpret_cast<const uint8_t*>(ptr) -
                         reinterpret_cast<const uint8_t*>(mmapped_buffer_);

    return static_cast<size_t>(signed_offset);
  }

  ANeuralNetworksMemory* memory() const { return handle_; }
  bool valid() const override { return handle_ != nullptr; }

 private:
  mutable ANeuralNetworksMemory* handle_ = nullptr;
};

class NNAPIDelegate {
 public:
  ~NNAPIDelegate();

  // Convert a tflite graph to NNAPI
  TfLiteStatus BuildGraph(::tflite::Subgraph* subgraph);

  // Run
  TfLiteStatus Invoke(::tflite::Subgraph* subgraph);

  // Whether the current platform supports NNAPI delegation.
  static bool IsSupported();

 private:
  // The NN API model handle
  ANeuralNetworksModel* nn_model_ = nullptr;
  // The NN API compilation handle
  ANeuralNetworksCompilation* nn_compiled_model_ = nullptr;
  // Model status
  TfLiteStatus model_status_ = kTfLiteOk;

  // List of state tensors for LSTM, RNN, SVDF.
  // NN API does not allow ops to maintain states across multiple
  // invocations. We need to manually create state input tensors from
  // corresponding state output tensors of TFLite operations, and map them
  // correctly.
  std::vector<int> model_states_inputs_;   // holds NNAPI operand ids
  std::vector<int> model_states_outputs_;  // holds TFLite tensor ids
};

} // namespace tflite
} // namespace nnfw

#endif  // __NNFW_TFLITE_EXT_NNAPI_DELEGATE_H__

// clang-format on
