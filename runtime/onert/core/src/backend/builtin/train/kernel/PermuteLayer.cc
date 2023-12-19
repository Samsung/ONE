

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
                           const std::vector<ITensor *> &input_back_prop_tensors,
                           const std::vector<ITensor *> &output_back_prop_tensors,
                           bool ignore_forward_in_training,
                           const std::shared_ptr<ExternalContext> &external_context)
  : builtin::kernel::PermuteLayer{src_tensors, dst_tensors, external_context},
    _input_back_prop_tensors{input_back_prop_tensors},
    _output_back_prop_tensors{output_back_prop_tensors},
    _ignore_forward_in_training{ignore_forward_in_training}
{
  assert(input_back_prop_tensors.size() == output_back_prop_tensors.size());
  assert(src_tensors.size() == dst_tensors.size());
}

void PermuteLayer::optimize()
{
  builtin::kernel::PermuteLayer::optimize();

  // TODO Calculate offsets of back propagation tensors if necessary
}

void PermuteLayer::forward(bool training)
{
  if (training && _ignore_forward_in_training)
    return;

  builtin::kernel::PermuteLayer::run();
}

void PermuteLayer::backward()
{
  for (uint32_t i = 0; i < _output_back_prop_tensors.size(); ++i)
  {
    auto src_back_prop = _output_back_prop_tensors.at(i);
    auto dst_back_prop = _input_back_prop_tensors.at(i);

    // NOTE The back propagation tensors corresponding to inputs/outputs of model are nullptr
    //      because permuting those tensors is meaningless
    if (src_back_prop && dst_back_prop)
    {
      const auto rank = src_back_prop->getShape().rank();
      auto output_offsets = _dst_tensors_offsets.at(i);
      auto input_offsets = _src_tensors_offsets.at(i);

      exec::IPermuteFunction::permute(src_back_prop, dst_back_prop, rank, output_offsets,
                                      input_offsets);
    }
  }
}

} // namespace kernel
} // namespace train
} // namespace builtin
} // namespace backend
} // namespace onert
