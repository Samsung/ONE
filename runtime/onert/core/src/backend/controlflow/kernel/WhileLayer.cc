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

#include "WhileLayer.h"

namespace onert
{
namespace backend
{
namespace controlflow
{
namespace kernel
{

WhileLayer::WhileLayer(std::vector<std::shared_ptr<backend::ITensor>> input_tensors,
                       std::vector<std::shared_ptr<backend::ITensor>> output_tensors,
                       exec::IExecutor &cond_executor, exec::IExecutor &body_executor)
    : _cond_executor{cond_executor}, _body_executor{body_executor}
{
  _src_tensors = input_tensors;
  _dst_tensors = output_tensors;
  for (size_t i = 0; i < input_tensors.size(); ++i)
  {
    auto rank = input_tensors.at(i)->num_dimensions();
    // TODO Remove this when applying dynamic tensor
    assert(rank == output_tensors.at(i)->num_dimensions());
    _ranks.emplace_back(rank);
  }
}

void WhileLayer::run()
{
  // TODO Implement like below comments
  // Copy _src_tensors -> inputs of cond subg
  // Run cond subg
  // Start loop while output of cond subg is ture
  // Copy cond subg inputs -> body subg inputs
  // Run body subg
  // Copy body subg outputs -> cond subg inputs
  // Run cond subg
  // Copy cond subg inputs -> _dst_tensors
}

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert
