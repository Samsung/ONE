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

#include "IfLayer.h"

#include <backend/ITensor.h>
#include "exec/ExecutorBase.h"

namespace onert
{
namespace backend
{
namespace controlflow
{
namespace kernel
{

IfLayer::IfLayer(const std::shared_ptr<backend::ITensor> &cond_tensor,
                 std::vector<std::shared_ptr<backend::ITensor>> input_tensors,
                 std::vector<std::shared_ptr<backend::ITensor>> output_tensors,
                 const ir::SubgraphIndex &then_subg_index, const ir::SubgraphIndex &else_subg_index,
                 const std::shared_ptr<exec::ExecutorMap> &executor_map)
    : _cond_tensor{cond_tensor}, _input_tensors{input_tensors}, _output_tensors{output_tensors},
      _then_subg_index{then_subg_index}, _else_subg_index{else_subg_index},
      _executor_map{executor_map}
{
  // At this point, executor_map may not have executors of then subg and else subg
}

void IfLayer::run()
{
  // TODO Support dynamic tensor
  // Check condition
  // // Copy _src_tensors -> then subg's inputs if true
  // // Copy _src_tensors -> else subg's inputs if false
  // Copy outputs of then subg or else subg -> _dst_tensors
}

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert
