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

#ifndef __ONERT_EXPORTER_TRAININFO_BUILDER_H__
#define __ONERT_EXPORTER_TRAININFO_BUILDER_H__

#include "ir/train/TrainingInfo.h"
#include "circle_schema_generated.h"
#include "circle_traininfo_generated.h"

namespace onert
{
namespace exporter
{

class TrainInfoBuilder
{
public:
  TrainInfoBuilder(const std::unique_ptr<ir::train::TrainingInfo> &training_info) : _builder(1024)
  {
    const auto optimizerInfo = training_info->optimizerInfo();
    const auto lossInfo = training_info->lossInfo();

    ::circle::Optimizer optimizer;
    ::circle::OptimizerOptions optimizer_opt_type;
    ::flatbuffers::Offset<void> optimizer_opt;
    switch (optimizerInfo.optim_code)
    {
      case ir::train::OptimizerCode::SGD:
        optimizer = ::circle::Optimizer_SGD;
        optimizer_opt_type = ::circle::OptimizerOptions_SGDOptions;
        optimizer_opt = ::circle::CreateSGDOptions(_builder, optimizerInfo.learning_rate).Union();
        break;
      case ir::train::OptimizerCode::Adam:
        optimizer = ::circle::Optimizer_ADAM;
        optimizer_opt_type = ::circle::OptimizerOptions_AdamOptions;
        optimizer_opt = ::circle::CreateAdamOptions(_builder, optimizerInfo.learning_rate).Union();
        break;
      default:
        throw std::runtime_error("Not supported optimizer code");
    }

    ::circle::LossFn lossfn;
    ::circle::LossFnOptions lossfn_opt_type;
    ::flatbuffers::Offset<void> lossfn_opt;
    switch (lossInfo.loss_code)
    {
      case ir::train::LossCode::MeanSquaredError:
        lossfn = ::circle::LossFn_MEAN_SQUARED_ERROR;
        lossfn_opt_type = ::circle::LossFnOptions_MeanSquaredErrorOptions;
        lossfn_opt = ::circle::CreateMeanSquaredErrorOptions(_builder).Union();
        break;
      case ir::train::LossCode::CategoricalCrossentropy:
        lossfn = ::circle::LossFn_CATEGORICAL_CROSSENTROPY;
        lossfn_opt_type = ::circle::LossFnOptions_CategoricalCrossentropyOptions;
        lossfn_opt = ::circle::CreateCategoricalCrossentropyOptions(_builder).Union();
        break;
      default:
        throw std::runtime_error("Not supported loss code");
    }

    ::circle::LossReductionType loss_reduction_type;
    switch (lossInfo.reduction_type)
    {
      case ir::train::LossReductionType::SumOverBatchSize:
        loss_reduction_type = ::circle::LossReductionType_SumOverBatchSize;
        break;
      case ir::train::LossReductionType::Sum:
        loss_reduction_type = ::circle::LossReductionType_Sum;
        break;
      default:
        throw std::runtime_error("Not supported loss reduction type");
    }

    std::vector<int32_t> trainable_ops;
    for (const auto &op : training_info->getTrainableOps())
    {
      trainable_ops.push_back(op.value());
    }

    const auto end = ::circle::CreateModelTrainingDirect(
      _builder, training_info->version(), optimizer, optimizer_opt_type, optimizer_opt, lossfn,
      lossfn_opt_type, lossfn_opt, 0, training_info->batchSize(), loss_reduction_type,
      &trainable_ops);
    _builder.Finish(end, ::circle::ModelTrainingIdentifier());

    ::flatbuffers::Verifier v(_builder.GetBufferPointer(), _builder.GetSize());
    bool verified = ::circle::VerifyModelTrainingBuffer(v);
    if (not verified)
      throw std::runtime_error{"TrainingInfo buffer is not accessible"};
  }

  uint8_t *get() const { return _builder.GetBufferPointer(); }
  uint32_t size() const { return _builder.GetSize(); }

private:
  ::flatbuffers::FlatBufferBuilder _builder;
};

} // namespace exporter
} // namespace onert

#endif // __ONERT_EXPORTER_TRAININFO_BUILDER_H__
