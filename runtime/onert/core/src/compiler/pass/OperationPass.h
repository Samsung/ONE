/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * @file  OperationPass.h
 * @brief This file contains OperationPass class
 */

#ifndef __ONERT_COMPILER_PASS_OPERATION_PASS_H__
#define __ONERT_COMPILER_PASS_OPERATION_PASS_H__

#include "Pass.h"
#include "ir/Index.h"

namespace onert
{
namespace ir
{
struct IOperation;
} // namespace ir
} // namespace onert

namespace onert
{
namespace compiler
{
namespace pass
{

/**
 * @brief  Class to iterate over operations and calls callback() method
 */
class OperationPass : public Pass
{
public:
  using Pass::Pass;
  virtual ~OperationPass() = default;

public:
  /**
   * @brief Returns string id for this pass. Same with class name.
   *
   * @return string id
   */
  std::string id() override = 0;

  /**
   * @brief Be called for all nodes of graph.
   * @param index is the index of a node in graph
   * @param node is the node in graph
   */
  virtual void callback(const ir::OperationIndex &index, ir::IOperation &node) = 0;

  /**
   * @brief Run the pass
   */
  void run() final;
};

} // namespace pass
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_PASS_OPERATION_PASS_H__
