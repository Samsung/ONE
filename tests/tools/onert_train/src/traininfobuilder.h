/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_TRAIN_TRAININFO_BUILDER_H__
#define __ONERT_TRAIN_TRAININFO_BUILDER_H__

#include <nnfw_experimental.h>
#include "circle_schema_generated.h"
#include "circle_traininfo_generated.h"

namespace onert_train
{

class TrainInfoBuilder
{
public:
  TrainInfoBuilder(const nnfw_train_info &traininfo) : _builder(1024)
  {
    ::circle::Optimizer optimizer;
    ::circle::OptimizerOptions optimizer_opt_type;
    ::flatbuffers::Offset<void> optimizer_opt;
    switch (traininfo.opt)
    {
      case NNFW_TRAIN_OPTIMIZER_SGD:
        optimizer = ::circle::Optimizer_SGD;
        optimizer_opt_type = ::circle::OptimizerOptions_SGDOptions;
        optimizer_opt = ::circle::CreateSGDOptions(_builder, traininfo.learning_rate).Union();
        break;
      case NNFW_TRAIN_OPTIMIZER_ADAM:
        optimizer = ::circle::Optimizer_ADAM;
        optimizer_opt_type = ::circle::OptimizerOptions_AdamOptions;
        optimizer_opt = ::circle::CreateAdamOptions(_builder, traininfo.learning_rate).Union();
        break;
      default:
        throw std::runtime_error("Not supported optimizer code");
    }

    ::circle::LossFn lossfn;
    ::circle::LossFnOptions lossfn_opt_type;
    ::flatbuffers::Offset<void> lossfn_opt;
    switch (traininfo.loss_info.loss)
    {
      case NNFW_TRAIN_LOSS_MEAN_SQUARED_ERROR:
        lossfn = ::circle::LossFn_MEAN_SQUARED_ERROR;
        lossfn_opt_type = ::circle::LossFnOptions_MeanSquaredErrorOptions;
        lossfn_opt = ::circle::CreateMeanSquaredErrorOptions(_builder).Union();
        break;
      case NNFW_TRAIN_LOSS_CATEGORICAL_CROSSENTROPY:
        lossfn = ::circle::LossFn_CATEGORICAL_CROSSENTROPY;
        lossfn_opt_type = ::circle::LossFnOptions_CategoricalCrossentropyOptions;
        lossfn_opt = ::circle::CreateCategoricalCrossentropyOptions(_builder).Union();
        break;
      default:
        throw std::runtime_error("Not supported loss code");
    }

    ::circle::LossReductionType loss_reduction_type;
    switch (traininfo.loss_info.reduction_type)
    {
      case NNFW_TRAIN_LOSS_REDUCTION_SUM_OVER_BATCH_SIZE:
        loss_reduction_type = ::circle::LossReductionType_SumOverBatchSize;
        break;
      case NNFW_TRAIN_LOSS_REDUCTION_SUM:
        loss_reduction_type = ::circle::LossReductionType_Sum;
        break;
      default:
        throw std::runtime_error("Not supported loss reduction type");
    }

    const auto end = ::circle::CreateModelTrainingDirect(
      _builder, 0, optimizer, optimizer_opt_type, optimizer_opt, lossfn, lossfn_opt_type,
      lossfn_opt, 0, traininfo.batch_size, loss_reduction_type, nullptr);
    _builder.Finish(end, ::circle::ModelTrainingIdentifier());

    ::flatbuffers::Verifier v(_builder.GetBufferPointer(), _builder.GetSize());
    bool verified = ::circle::VerifyModelTrainingBuffer(v);
    if (not verified)
      throw std::runtime_error{"TrainingInfo buffer is not accessible"};
  }

  uint8_t *get() const { return _builder.GetCurrentBufferPointer(); }
  uint32_t size() const { return _builder.GetSize(); }

private:
  ::flatbuffers::FlatBufferBuilder _builder;
};

} // namespace onert_train

#endif // __ONERT_TRAIN_TRAININFO_BUILDER_H__
