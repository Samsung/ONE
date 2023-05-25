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
#include "exec/ITrainableFunction.h"
#include "ir/train/ITrainableOperation.h"

namespace onert
{
namespace compiler
{
namespace train
{

// TODO Find a better way instead of inheriting compiler::CodeAndInfo
struct CodeAndInfo : public compiler::CodeAndInfo
{
  std::unique_ptr<exec::ITrainableFunction> trainable_fn;
  const ir::train::ITrainableOperation *op;

  CodeAndInfo(const ir::OperationIndex op_ind, const ir::train::ITrainableOperation *op,
              const OperationLowerInfo *lower_info,
              std::unique_ptr<exec::ITrainableFunction> &&trainable_fn)
    : compiler::CodeAndInfo{op_ind, &(op->operation()), lower_info,
                            std::make_unique<exec::FunctionSequence>()},
      trainable_fn{std::move(trainable_fn)}, op{op}
  {
  }
};

} // namespace train
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_TRAIN_CODE_MAP_H__
