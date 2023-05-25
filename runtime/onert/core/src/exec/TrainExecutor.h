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

/**
 * @file  TrainExecutor.h
 * @brief This file contains TrainExecutor class to define and run execution phase
 */

#ifndef __ONERT_EXEC_TRAIN_EXECUTOR_H_
#define __ONERT_EXEC_TRAIN_EXECUTOR_H_

#include "ExecutorBase.h"

#include "compiler/CodeMap.h"
#include "ir/Index.h"
#include "util/TracingCtx.h"

namespace onert
{
namespace exec
{

/**
 * @brief Class to handle execution phase. Simple run the sequence of operations that is sorted in
 *        topological order
 */
class TrainExecutor final : public ExecutorBase
{
public:
  /**
   * @brief Construct a new TrainExecutor object
   * @param lowered_graph LoweredGraph object
   * @param tensor_builders Tensor builders that are currently used
   * @param code_map @c ir::Operation and its code map
   */
  TrainExecutor(std::unique_ptr<compiler::LoweredGraph> lowered_graph,
                 backend::BackendContexts &&backend_contexts,
                 const compiler::TensorRegistries &tensor_regs, compiler::CodeMap &&code_map,
                 const std::vector<ir::OperationIndex> &order, const util::TracingCtx *tracing_ctx)
    : ExecutorBase{std::move(lowered_graph), std::move(backend_contexts), tensor_regs, tracing_ctx}
  {
    for (auto index : order)
    {
      _code.emplace_back(std::move(code_map.at(index)));
    }
  }

public:
  void executeImpl(void) override;

private:
  std::vector<compiler::CodeAndInfo> _code;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_TRAIN_EXECUTOR_H_
