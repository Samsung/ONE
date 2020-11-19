/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * @file  LinearExecutor.h
 * @brief This file contains LinearExecutor class to define and run execution phase
 */

#ifndef __ONERT_EXEC_EXECUTOR_H_
#define __ONERT_EXEC_EXECUTOR_H_

#include "ir/Index.h"
#include "ExecutorBase.h"
#include "compiler/Linear.h"
#include "exec/FunctionSequence.h"
#include "compiler/CodeMap.h"

namespace onert
{
namespace exec
{

/**
 * @brief Class to handle execution phase. Simple run the sequence of operations that is sorted in
 *        topological order
 */
class LinearExecutor final : public ExecutorBase
{
public:
  /**
   * @brief Construct a new LinearExecutor object
   * @param lowered_graph LoweredGraph object
   * @param tensor_builders Tensor builders that are currently used
   * @param code_map OpSequence and its code map
   */
  LinearExecutor(std::unique_ptr<compiler::LoweredGraph> lowered_graph,
                 const compiler::TensorRegistries &tensor_regs, compiler::CodeMap &&code_map,
                 const std::vector<ir::OpSequenceIndex> &order)
      : ExecutorBase{std::move(lowered_graph), tensor_regs}
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

#endif // __ONERT_EXEC_EXECUTOR_H_
