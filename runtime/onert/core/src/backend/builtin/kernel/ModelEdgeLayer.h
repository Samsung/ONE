/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_BUILTIN_KERNEL_MODEL_EDGE_LAYER_H__
#define __ONERT_BACKEND_BUILTIN_KERNEL_MODEL_EDGE_LAYER_H__

#include <backend/ITensor.h>
#include <exec/Executors.h>
#include "../ExternalContext.h"

namespace onert
{
namespace backend
{
namespace builtin
{
namespace kernel
{

class ModelEdgeLayer : public ::onert::exec::IFunction
{
public:
  ModelEdgeLayer(const std::vector<backend::ITensor *> input_tensors,
                 const ir::ModelIndex &to_model_index, const ir::SubgraphIndex &to_subg_index,
                 const ir::IOIndex &to_io_index, exec::Executors *executors,
                 const std::shared_ptr<ExternalContext> &external_context);

public:
  void run() override;

private:
  const std::vector<backend::ITensor *> _input_tensors;
  const ir::ModelIndex _to_model_index;
  const ir::SubgraphIndex _to_subg_index;
  const ir::IOIndex _to_io_index;
  exec::Executors *_executors;
  const std::shared_ptr<ExternalContext> _external_context;
};

} // namespace kernel
} // namespace builtin
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BUILTIN_KERNEL_MODEL_EDGE_LAYER_H__
