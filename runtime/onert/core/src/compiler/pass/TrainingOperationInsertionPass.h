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
 * @file  TrainingOperationInsertionPass.h
 * @brief This file contains TrainingOperationInsertionPass class
 */

#ifndef __ONERT_COMPILER_PASS_TRAINING_OPERATION_INSERTION_PASS_H__
#define __ONERT_COMPILER_PASS_TRAINING_OPERATION_INSERTION_PASS_H__

#include "Pass.h"
#include "ir/TrainingInfo.h"

namespace onert
{
namespace compiler
{
namespace pass
{

/**
 * @brief  A pass to insert training operation into the graph
 *
 * Find operations to be inserted from training information, generate and insert the operations.
 *
 */
class TrainingOperationInsertionPass : public Pass
{
public:
  TrainingOperationInsertionPass(ir::Graph &graph, const ir::TrainingInfo *training_info)
    : Pass{graph}, _training_info{training_info}
  {
  }

public:
  std::string id() override { return "TrainingOperationInsertionPass"; }
  void run() final;

private:
  const ir::TrainingInfo *_training_info;
};

} // namespace pass
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_PASS_TRAINING_OPERATION_INSERTION_PASS_H__
