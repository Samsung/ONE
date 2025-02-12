/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAIN_TENSOR_PLANNER_H__
#define __ONERT_BACKEND_TRAIN_TENSOR_PLANNER_H__

#include "TensorBuilder.h"

#include <ir/train/TrainableGraph.h>
#include <util/Set.h>

namespace onert::backend::train
{

class TensorPlanner
{
public:
  TensorPlanner(const ir::train::TrainableGraph &tgraph,
                const util::Set<ir::OperandIndex> &external_operands);
  TensorPlanner(const TensorPlanner &) = delete;
  TensorPlanner(TensorPlanner &&) = delete;
  TensorPlanner &operator=(const TensorPlanner &) = delete;
  TensorPlanner &operator=(TensorPlanner &&) = delete;
  ~TensorPlanner() = default;

  void planNonConstTensors(TensorBuilder *tensor_builder);
  void planTrainableTensors(TensorBuilder *tensor_builder);
  void planBackPropTensors(TensorBuilder *tensor_builder);
  void planGradientTensors(TensorBuilder *tensor_builder);
  void planDisposableBackPropTensors(TensorBuilder *tensor_builder);
  void planLayerScopeTensors(TensorBuilder *tensor_builder);

private:
  ir::OperandIndexSequence getOutgoingBackPropSeq(const ir::OperationIndex &op_index,
                                                  const TensorBuilder *tensor_builder);

private:
  const ir::train::TrainableGraph &_tgraph;
  const util::Set<ir::OperandIndex> &_external_operands;
};

} // namespace onert::backend::train

#endif // __ONERT_BACKEND_TRAIN_TENSOR_PLANNER_H__
