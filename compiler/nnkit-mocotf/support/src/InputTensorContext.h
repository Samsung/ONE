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

#ifndef __NNKIT_SUPPORT_MOCO_TF_INPUT_TENSOR_CONTEXT_H__
#define __NNKIT_SUPPORT_MOCO_TF_INPUT_TENSOR_CONTEXT_H__

#include "TensorContext.h"

#include <nnkit/TensorContext.h>
#include <nnkit/support/tftestinfo/ParsedTensor.h>

#include <locomotiv/NodeData.h>

#include <vector>
#include <memory>

namespace nnkit
{
namespace support
{
namespace moco
{
namespace tf
{

/**
 * @brief Class for the context of input tensors
 */
class InputTensorContext final : public TensorContext
{
  using Buffers = std::vector<std::unique_ptr<nncc::core::ADT::tensor::Buffer<float>>>;

public:
  InputTensorContext(const ParsedTensors &parsed_tensors, const Buffers &buffers)
    : TensorContext(parsed_tensors), _buffers(buffers)
  { /* empty */
  }

  InputTensorContext(const InputTensorContext &) = delete;
  InputTensorContext(InputTensorContext &&) = delete;

public:
  void getMutableFloatTensor(uint32_t n,
                             const nnkit::TensorContext::TypedAccessor<float> &f) override;

  void getConstFloatTensor(uint32_t n,
                           const nnkit::TensorContext::TypedReader<float> &f) const override;

private:
  const Buffers &_buffers;
};

} // namespace tf
} // namespace moco
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_MOCO_TF_INPUT_TENSOR_CONTEXT_H__
