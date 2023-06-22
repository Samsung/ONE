

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
                           const std::shared_ptr<ExternalContext> &external_context)
  : builtin::kernel::PermuteLayer{src_tensors, dst_tensors, external_context}
{
}

void PermuteLayer::forward(bool) { builtin::kernel::PermuteLayer::run(); }

} // namespace kernel
} // namespace train
} // namespace builtin
} // namespace backend
} // namespace onert
