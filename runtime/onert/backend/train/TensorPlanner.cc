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

#include "TensorPlanner.h"

#include <util/logging.h>

namespace onert
{
namespace backend
{
namespace train
{

TensorPlanner::TensorPlanner(const ir::train::TrainableGraph &tgraph,
                             const util::Set<ir::OperandIndex> &external_operands)
  : _tgraph{tgraph}, _external_operands{external_operands}
{
  // DO NOTHING
}

void TensorPlanner::planNonConstTensors(TensorBuilder *tensor_builder)
{
  // TODO Plan non-const tensors
}

void TensorPlanner::planTrainableTensors(TensorBuilder *tensor_builder)
{
  // TODO Plan trainable tensors such as weights
}

void TensorPlanner::planBackPropTensors(TensorBuilder *tensor_builder)
{
  // TODO Plan back-propagated tensors
}

void TensorPlanner::planGradientTensors(TensorBuilder *tensor_builder)
{
  // TODO Plan gradient tensors
}

void TensorPlanner::planDisposableBackPropTensors(TensorBuilder *tensor_builder)
{
  // TODO Plan diposable backprop tensors
}

} // namespace train
} // namespace backend
} // namespace onert
