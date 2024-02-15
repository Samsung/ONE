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

CirclePlusBuffer CirclePlusGen::finish()
{
  CirclePlusBuffer cpbuf;
  cpbuf.circle = CircleGen::finish();
  cpbuf.circle_plus = createModelTraining();
  return cpbuf;
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

  // NOTE: epochs will be removed
  auto model_training =
    circle::CreateModelTraining(_fbb_plus, 0, optimizer, optimizer_opt_type, optimizer_opt, lossfn,
                                lossfn_opt_type, lossfn_opt, 0, batch_size, loss_reduction_type);
  _fbb_plus.Finish(model_training, circle::ModelTrainingIdentifier());

  auto cpbuf = CircleBuffer{std::move(_fbb_plus)};
  {
    // For the only this draft
    // Verify model. Code is copied from runtime/onert/core/src/loader/traininfo_loader.cc
    const uint8_t *buffer = cpbuf.buffer();
    const size_t size = cpbuf.size();
    assert(buffer != nullptr);
    assert(size > 0);
    assert(circle::ModelTrainingBufferHasIdentifier(buffer));
    flatbuffers::Verifier v(buffer, size);
    assert(circle::VerifyModelTrainingBuffer(v));
    (void)v;
  }
  return cpbuf;
}
