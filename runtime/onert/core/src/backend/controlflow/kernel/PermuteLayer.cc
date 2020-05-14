/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
namespace controlflow
{
namespace kernel
{

void PermuteLayer::configure(std::shared_ptr<backend::ITensor> input,
                             std::shared_ptr<backend::ITensor> output, size_t rank)
{
  _src_tensors.emplace_back(input);
  _dst_tensors.emplace_back(output);
  _ranks.emplace_back(rank);
}

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert
