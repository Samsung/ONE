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
 * @file  InterpExecutor.h
 * @brief This file contains InterpExecutor class\n
 *        to manage interpreter execution and environment
 */
#ifndef __ONERT_INTERP_INTERP_EXECUTOR_H__
#define __ONERT_INTERP_INTERP_EXECUTOR_H__

#include "ir/OperandIndexMap.h"
#include "ir/Graph.h"
#include "exec/IExecutor.h"

namespace onert
{
namespace interp
{

class ITensor;

/**
 * @brief Class to execute model using interpreter
 */
class InterpExecutor final : public exec::IExecutor
{
public:
  explicit InterpExecutor(const ir::Graph &graph) : _graph(graph), _execution_done(false)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief   Return graph object
   * @return  Graph object
   */
  const ir::Graph &graph() final { return _graph; }
  void setIndexedRanks(std::shared_ptr<ir::OperationIndexMap<int64_t>>) override{
      // Not implemented
  };
  /**
   * @brief  Start execution
   * @note   It should be called after setting input and output buffer
   */
  void execute(const exec::IODescription &desc) final;

  void fillOutputShapes(std::unordered_map<ir::IOIndex, ir::Shape> *output_shapes) override;

private:
  const ir::Graph &_graph;
  ir::OperandIndexMap<std::shared_ptr<ITensor>> _tensor_map;
  bool _execution_done;
};

} // namespace interp
} // namespace onert

#endif // __ONERT_INTERP_INTERP_EXECUTOR_H__
