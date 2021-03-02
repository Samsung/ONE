/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "nnkit/support/onnx/Runner.h"
#include "nnkit/support/onnx/Status.h"

#include <memory>
#include <cassert>

namespace nnkit
{
namespace support
{
namespace onnx
{

Runner::Runner(const std::string &path) : _allocator(std::make_unique<Allocator>())
{
  Status status;

  status = OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "nnkit", &_env);
  assert(!status.isError());

  status = OrtCreateSession(_env, path.c_str(), nullptr, &_session);
  assert(!status.isError());
}

Runner::~Runner(void)
{
  if (_session)
  {
    OrtReleaseSession(_session);
  }

  if (_env)
  {
    OrtReleaseEnv(_env);
  }
}

void Runner::prepareInputs(void)
{
  Status status;

  assert(_inputs == nullptr);

  size_t num_input_nodes;
  status = OrtSessionGetInputCount(_session, &num_input_nodes);
  status.throwOnError();

  _inputs = std::make_unique<TensorSet>(_allocator.get(), num_input_nodes);

  for (size_t i = 0; i < num_input_nodes; ++i)
  {
    char *input_name;
    status = OrtSessionGetInputName(_session, i, _allocator.get(), &input_name);
    status.throwOnError();

    assert(input_name != nullptr);

    std::string name{input_name};
    _allocator->Free(input_name);

    OrtTypeInfo *typeinfo;
    status = OrtSessionGetInputTypeInfo(_session, i, &typeinfo);
    status.throwOnError();

    const OrtTensorTypeAndShapeInfo *tensor_info = OrtCastTypeInfoToTensorInfo(typeinfo);
    ONNXTensorElementDataType type = OrtGetTensorElementType(tensor_info);

    uint32_t num_dims = OrtGetNumOfDimensions(tensor_info);
    std::vector<size_t> dims(num_dims);
    OrtGetDimensions(tensor_info, (int64_t *)dims.data(), num_dims);

    // NOTE To run OnnxRuntime, the total size of input tensor must be fixed.
    //      In the present code, the unknown shape that is -1 is arbitrarily changed to 1.
    //
    // TODO Add user argument related to unknown shape
    //
    for (uint32_t j = 0; j < num_dims; ++j)
    {
      if (dims[j] == -1)
      {
        dims[j] = 1;
      }
    }
    OrtReleaseTypeInfo(typeinfo);

    _inputs->set(i, name, type, dims);
  }
}

void Runner::prepareOutputs(void)
{
  Status status;

  assert(_outputs == nullptr);

  size_t num_output_nodes;
  status = OrtSessionGetOutputCount(_session, &num_output_nodes);
  status.throwOnError();

  _outputs = std::make_unique<TensorSet>(_allocator.get(), num_output_nodes);

  for (size_t i = 0; i < num_output_nodes; ++i)
  {
    char *output_name;
    status = OrtSessionGetOutputName(_session, i, _allocator.get(), &output_name);
    status.throwOnError();

    assert(output_name != nullptr);

    std::string name{output_name};
    _allocator->Free(output_name);

    OrtTypeInfo *typeinfo;
    status = OrtSessionGetOutputTypeInfo(_session, i, &typeinfo);
    status.throwOnError();

    const OrtTensorTypeAndShapeInfo *tensor_info = OrtCastTypeInfoToTensorInfo(typeinfo);
    ONNXTensorElementDataType type = OrtGetTensorElementType(tensor_info);

    uint32_t num_dims = OrtGetNumOfDimensions(tensor_info);
    std::vector<size_t> dims(num_dims);
    OrtGetDimensions(tensor_info, (int64_t *)dims.data(), num_dims);

    // NOTE To run OnnxRuntime, the total size of output tensor must be fixed.
    //      In the present code, the unknown shape that is -1 is arbitrarily changed to 1.
    //
    // TODO Add user argument related to unknown shape
    //
    for (uint32_t j = 0; j < num_dims; ++j)
    {
      if (dims[j] == -1)
      {
        dims[j] = 1;
      }
    }
    OrtReleaseTypeInfo(typeinfo);

    _outputs->set(i, name, type, dims);
  }
}

void Runner::run(void)
{
  Status status;

  auto pinput_names = _inputs->names();
  std::vector<const char *> input_names(pinput_names.size());
  for (size_t i = 0; i < pinput_names.size(); ++i)
  {
    input_names[i] = pinput_names[i].c_str();
  }

  auto poutput_names = _outputs->names();
  std::vector<const char *> output_names(poutput_names.size());
  for (size_t i = 0; i < poutput_names.size(); ++i)
  {
    output_names[i] = poutput_names[i].c_str();
  }

  status = OrtRun(_session, NULL, input_names.data(), _inputs->tensors().data(), _inputs->size(),
                  output_names.data(), _outputs->size(), _outputs->mutable_tensors().data());
  status.throwOnError();
}

} // namespace onnx
} // namespace support
} // namespace nnkit
