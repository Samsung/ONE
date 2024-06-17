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

#ifndef __ONERT_BACKEND_BASIC_TRAIN_TRAINABLE_BACKEND_CONTEXT_HELPERS_H__
#define __ONERT_BACKEND_BASIC_TRAIN_TRAINABLE_BACKEND_CONTEXT_HELPERS_H__

#include "backend/basic/BackendContextHelpers.h"
#include "backend/train/TrainableBackendContext.h"

namespace onert
{
namespace backend
{
namespace basic
{
namespace train
{

// TODO Unify with the above `getTensors()` function in `BackendContextHelpers.h`
template <typename TensorBuilder>
ITensorRegistry *genTensors(backend::train::TrainableBackendContext &ctx,
                            const std::shared_ptr<TensorBuilder> &tensor_builder)
{
  const auto &tgraph = *ctx.trainable_graph();

  tgraph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &obj) {
    if (ctx.external_operands().contains(ind))
      return;
    // NOTE Assuming there is no layout changes (Always assume NHWC or UNKNOWN)
    assert(tgraph.layout() != ir::Layout::NCHW);
    tensor_builder->registerTensorInfo(ind, obj.info(), ir::Layout::NHWC);
  });

  // For the executors that does not have fixed linear execution order:
  // To make tensors never be deallocated, this is a workaround to use static memory planner
  tgraph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &) {
    if (tensor_builder->isRegistered(ind))
      tensor_builder->notifyFirstUse(ind);
  });

  tensor_builder->allocate();

  return ctx.tensor_registry().get();
}

} // namespace train
} // namespace basic
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BASIC_TRAIN_TRAINABLE_BACKEND_CONTEXT_HELPERS_H__
