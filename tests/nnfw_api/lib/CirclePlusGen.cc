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

#include "CirclePlusGen.h"

#include <numeric>
#include <nnfw_experimental.h> // for NNFW_TRAIN_NUM_OF_TRAINABLE_OPS_SPECIAL_VALUES

CircleBuffers CirclePlusGen::finish()
{
  CircleBuffers cbufs;
  cbufs.circle = CircleGen::finish();
  cbufs.circle_plus = createModelTraining();
  return cbufs;
}

void CirclePlusGen::addTrainInfo(const TrainInfo &info) { _info = info; }

CircleBuffer CirclePlusGen::createModelTraining()
{
  circle::OptimizerOptions optimizer_opt_type = circle::OptimizerOptions::OptimizerOptions_NONE;
  flatbuffers::Offset<void> optimizer_opt = 0;
  const circle::Optimizer optimizer = _info.optimizer;
  switch (optimizer)
  {
    case circle::Optimizer_SGD:
      optimizer_opt_type = circle::OptimizerOptions::OptimizerOptions_SGDOptions;
      optimizer_opt = circle::CreateSGDOptions(_fbb_plus, _info.learning_rate).Union();
      break;
    case circle::Optimizer_ADAM:
      optimizer_opt_type = circle::OptimizerOptions::OptimizerOptions_AdamOptions;
      optimizer_opt = circle::CreateAdamOptions(_fbb_plus, _info.learning_rate).Union();
      break;
    default:
      throw std::runtime_error("unknown optimzer");
  }

  circle::LossFnOptions lossfn_opt_type = circle::LossFnOptions_NONE;
  flatbuffers::Offset<void> lossfn_opt = 0;
  const circle::LossFn lossfn = _info.loss_fn;
  switch (lossfn)
  {
    case circle::LossFn::LossFn_CATEGORICAL_CROSSENTROPY:
      lossfn_opt_type = circle::LossFnOptions::LossFnOptions_CategoricalCrossentropyOptions;
      lossfn_opt = circle::CreateCategoricalCrossentropyOptions(_fbb_plus).Union();
      break;
    case circle::LossFn::LossFn_MEAN_SQUARED_ERROR:
      lossfn_opt_type = circle::LossFnOptions::LossFnOptions_MeanSquaredErrorOptions;
      lossfn_opt = circle::CreateMeanSquaredErrorOptions(_fbb_plus).Union();
      break;
    case circle::LossFn::LossFn_SPARSE_CATEGORICAL_CROSSENTROPY:
      // TODO enable this conversion after core support sparse_categorial_crossentropy
      throw std::runtime_error{"'sparse_categorical_crossentropy' is not supported yet"};
    default:
      throw std::runtime_error{"unknown loss function"};
  }

  circle::LossReductionType loss_reduction_type = _info.loss_reduction_type;

  int32_t batch_size = _info.batch_size;

  const uint32_t ops_size = getCurrentSubgraphOpsSize();
  std::vector<int32_t> trainable_ops;
  if (NNFW_TRAIN_TRAINABLE_ALL == _info.num_of_trainable_ops)
  {
    for (uint32_t idx = 0; idx < ops_size; ++idx)
    {
      trainable_ops.push_back(idx);
    }
  }
  else if (_info.num_of_trainable_ops > 0)
    for (uint32_t i = 1; i <= static_cast<uint32_t>(_info.num_of_trainable_ops); ++i)
    {
      trainable_ops.push_back(ops_size - i);
    }
  else if (_info.num_of_trainable_ops <= NNFW_TRAIN_TRAINABLE_INCORRECT_STATE)
  {
    throw std::invalid_argument("Incorrect negative value of num_of_trainable_ops");
  }

  // NOTE: epochs will be removed
  auto model_training = circle::CreateModelTraining(
    _fbb_plus, 0, optimizer, optimizer_opt_type, optimizer_opt, lossfn, lossfn_opt_type, lossfn_opt,
    0, batch_size, loss_reduction_type, _fbb_plus.CreateVector(trainable_ops));
  _fbb_plus.Finish(model_training, circle::ModelTrainingIdentifier());

  return CircleBuffer{std::move(_fbb_plus)};
}
