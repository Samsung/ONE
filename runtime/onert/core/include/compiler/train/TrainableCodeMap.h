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

#ifndef __ONERT_COMPILER_TRAIN_TRAINABLE_CODE_MAP_H__
#define __ONERT_COMPILER_TRAIN_TRAINABLE_CODE_MAP_H__

#include "backend/Backend.h"
#include "exec/train/TrainableFnSequence.h"
#include "ir/train/ITrainableOperation.h"

#include <unordered_map>

namespace onert::compiler::train
{

struct TrainableCodeAndInfo
{
  ir::OperationIndex op_ind;
  const ir::train::ITrainableOperation *op;
  const backend::Backend *op_backend;
  // TODO Change to TrainableFnSequence
  std::unique_ptr<exec::train::TrainableFnSequence> tn_seq;

  TrainableCodeAndInfo(const ir::OperationIndex op_ind, const ir::train::ITrainableOperation *op,
                       const backend::Backend *op_backend,
                       std::unique_ptr<exec::train::TrainableFnSequence> &&tn_seq)
    : op_ind{op_ind}, op{op}, op_backend{op_backend}, tn_seq{std::move(tn_seq)}
  {
  }
};

using TrainableCodeMap = std::unordered_map<ir::OperationIndex, TrainableCodeAndInfo>;

} // namespace onert::compiler::train

#endif // __ONERT_COMPILER_TRAIN_TRAINABLE_CODE_MAP_H__
