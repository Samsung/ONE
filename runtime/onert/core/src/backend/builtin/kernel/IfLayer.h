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

#ifndef __ONERT_BACKEND_BUILTIN_KERNEL_IF_LAYER_H__
#define __ONERT_BACKEND_BUILTIN_KERNEL_IF_LAYER_H__

#include <backend/IPortableTensor.h>
#include <exec/IExecutors.h>
#include <exec/IFunction.h>
#include "../ExternalContext.h"

namespace onert
{
namespace backend
{
namespace builtin
{
namespace kernel
{

class IfLayer : public ::onert::exec::IFunction
{
public:
  IfLayer(backend::IPortableTensor *cond_tensor,
          const std::vector<backend::IPortableTensor *> input_tensors,
          const std::vector<backend::IPortableTensor *> output_tensors,
          const ir::SubgraphIndex &then_subg_index, const ir::SubgraphIndex &else_subg_index,
          exec::IExecutors *executors, const ir::ModelIndex &model_index,
          const std::shared_ptr<ExternalContext> &external_context);

public:
  void run() override;

private:
  backend::IPortableTensor *_cond_tensor;
  const std::vector<backend::IPortableTensor *> _input_tensors;
  const std::vector<backend::IPortableTensor *> _output_tensors;
  const ir::SubgraphIndex _then_subg_index;
  const ir::SubgraphIndex _else_subg_index;
  exec::IExecutors *_executors;
  ir::ModelIndex _model_index;
  const std::shared_ptr<ExternalContext> _external_context;
};

} // namespace kernel
} // namespace builtin
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BUILTIN_KERNEL_IF_LAYER_H__
