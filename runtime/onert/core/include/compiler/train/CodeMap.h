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

#ifndef __ONERT_COMPILER_TRAIN_CODE_MAP_H__
#define __ONERT_COMPILER_TRAIN_CODE_MAP_H__

#include <unordered_map>
#include "compiler/CodeMap.h"
#include "exec/train/TrainableSequence.h"
#include "ir/train/ITrainableOperation.h"

namespace onert
{
namespace compiler
{
namespace train
{

struct CodeAndInfo
{
  ir::OperationIndex op_ind;
  const ir::train::ITrainableOperation *op;
  const OperationLowerInfo *lower_info;
  // TODO Change to TrainableSequence
  std::unique_ptr<exec::train::TrainableSequence> tn_seq;

  CodeAndInfo(const ir::OperationIndex op_ind, const ir::train::ITrainableOperation *op,
              const OperationLowerInfo *lower_info,
              std::unique_ptr<exec::train::TrainableSequence> &&tn_seq)
    : op_ind{op_ind}, op{op}, lower_info{lower_info}, tn_seq{std::move(tn_seq)}
  {
  }
};

using CodeMap = std::unordered_map<ir::OperationIndex, CodeAndInfo>;

} // namespace train
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_TRAIN_CODE_MAP_H__
