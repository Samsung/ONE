

/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "PermuteLayer.h"

namespace onert
{
namespace backend
{
namespace builtin
{
namespace train
{
namespace kernel
{

PermuteLayer::PermuteLayer(const std::vector<ITensor *> &src_tensors,
                           const std::vector<ITensor *> &dst_tensors,
                           const std::vector<ITensor *> &input_deriv_tensors,
                           const std::vector<ITensor *> &output_deriv_tensors,
                           const std::shared_ptr<ExternalContext> &external_context)
  : builtin::kernel::PermuteLayer{src_tensors, dst_tensors, external_context},
    _input_deriv_tensor{input_deriv_tensors}, _output_deriv_tensor{output_deriv_tensors}
{
  assert(_input_deriv_tensor.size() == output_deriv_tensors.size());

  optimize();
}

void PermuteLayer::forward(bool) { builtin::kernel::PermuteLayer::run(); }

void PermuteLayer::backward()
{
  for (uint32_t i = 0; i < _output_deriv_tensor.size(); ++i)
  {
    auto src_deriv = _output_deriv_tensor.at(i);
    auto dst_deriv = _input_deriv_tensor.at(i);
    const auto rank = src_deriv->getShape().rank();
    // TODO Calculate offsets of derivative tensors
    auto output_offsets = _dst_tensors_offsets.at(i);
    auto input_offsets = _src_tensors_offsets.at(i);

    exec::IPermuteFunction::permute(src_deriv, dst_deriv, rank, output_offsets, input_offsets);
  }
}

} // namespace kernel
} // namespace train
} // namespace builtin
} // namespace backend
} // namespace onert
