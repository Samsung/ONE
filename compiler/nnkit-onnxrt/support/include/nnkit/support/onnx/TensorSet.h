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

#ifndef __NNKIT_SUPPORT_ONNX_TENSOR_SET_H__
#define __NNKIT_SUPPORT_ONNX_TENSOR_SET_H__

#include "nnkit/support/onnx/Allocator.h"
#include "nnkit/support/onnx/Status.h"

#include <onnxruntime_c_api.h>

#include <string>
#include <vector>
#include <cassert>

namespace nnkit
{
namespace support
{
namespace onnx
{

class TensorSet final
{
public:
  TensorSet(Allocator *allocator, size_t nums)
    : _allocator(allocator), _names(nums), _types(nums), _dims(nums), _tensors(nums, nullptr)
  {
    // DO NOTHING
  }

  ~TensorSet(void)
  {
    for (auto it : _tensors)
    {
      OrtReleaseValue(it);
    }
  }

  void set(size_t index, const std::string &name, ONNXTensorElementDataType type,
           const std::vector<size_t> &dims)
  {
    _names[index] = name;
    _types[index] = type;
    _dims[index] = dims;

    Status status;

    status =
      OrtCreateTensorAsOrtValue(_allocator, dims.data(), dims.size(), type, &_tensors[index]);
    status.throwOnError();

    assert(OrtIsTensor(_tensors[index]));
  }

  size_t size(void) { return _names.size(); }

  const char *name(size_t index) { return _names[index].c_str(); }
  const std::vector<std::string> &names(void) { return _names; }

  ONNXTensorElementDataType type(size_t index) { return _types[index]; }

  const std::vector<size_t> &dim(size_t index) { return _dims[index]; }

  const OrtValue *tensor(size_t index) { return _tensors[index]; }
  const std::vector<OrtValue *> &tensors(void) { return _tensors; }

  OrtValue *mutable_tensor(size_t index) { return _tensors[index]; }
  std::vector<OrtValue *> mutable_tensors(void) { return _tensors; }

private:
  Allocator *_allocator;

  std::vector<std::string> _names;
  std::vector<ONNXTensorElementDataType> _types;
  std::vector<std::vector<size_t>> _dims;
  std::vector<OrtValue *> _tensors;
};

} // namespace onnx
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_ONNX_TENSOR_SET_H__
