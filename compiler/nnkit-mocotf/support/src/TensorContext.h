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

#ifndef __NNKIT_SUPPORT_MOCO_TF_TENSOR_CONTEXT_H__
#define __NNKIT_SUPPORT_MOCO_TF_TENSOR_CONTEXT_H__

#include "nnkit/TensorContext.h"
#include "nnkit/support/tftestinfo/ParsedTensor.h"

#include <nncc/core/ADT/tensor/Shape.h>

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
using nnkit::support::tftestinfo::ParsedTensor;
using ParsedTensors = std::vector<std::unique_ptr<ParsedTensor>>;

/**
 * @brief Parent class of InputTensorContext and OutputTensorContext
 */
class TensorContext : public nnkit::TensorContext
{
public:
  TensorContext(const ParsedTensors &tensors) : _tensors(tensors)
  {
    // empty
  }

  TensorContext(const TensorContext &) = delete;
  TensorContext(TensorContext &&) = delete;

public:
  uint32_t size(void) const override { return _tensors.size(); }

public:
  std::string name(uint32_t n) const override // name with ":0", ":1", etc
  {
    return _tensors.at(n)->name();
  }

public:
  nncc::core::ADT::tensor::Shape shape(uint32_t n) const override
  {
    return _tensors.at(n)->shape();
  }

public:
  // Float (fp32) tensor support
  bool isFloatTensor(uint32_t n) const override { return _tensors.at(n)->isFloatTensor(); }

  void getMutableFloatTensor(uint32_t n,
                             const nnkit::TensorContext::TypedAccessor<float> &f) override;
  void getConstFloatTensor(uint32_t n,
                           const nnkit::TensorContext::TypedReader<float> &f) const override;

private:
  const ParsedTensors &_tensors;
};

} // namespace tf
} // namespace moco
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_MOCO_TF_TENSOR_CONTEXT_H__
