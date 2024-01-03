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

#include "traininfo_loader.h"
#include "circle_traininfo_generated.h"
#include "flatbuffers/flatbuffers.h"

namespace onert
{
namespace train
{
namespace traininfo_loader
{

namespace
{

ir::train::OptimizerInfo loadOptimizerInfo(const circle::ModelTraining *circle_model)
{
  // fill ir_opt from cirlce_opt
  ir::train::OptimizerInfo ir_opt;
  const circle::Optimizer circle_opt = circle_model->optimizer();

  switch (circle_opt)
  {
    case circle::Optimizer_SGD:
      ir_opt.optim_code = ir::train::OptimizerCode::SGD;
      ir_opt.learning_rate = circle_model->optimizer_opt_as_SGDOptions()->learning_rate();
      break;
    case circle::Optimizer_ADAM:
      ir_opt.optim_code = ir::train::OptimizerCode::Adam;
      ir_opt.learning_rate = circle_model->optimizer_opt_as_AdamOptions()->learning_rate();
      break;
    default:
      throw std::runtime_error("unknown optimzer");
  }
  return ir_opt;
}

ir::train::LossInfo loadLossInfo(const circle::ModelTraining *circle_model)
{
  // fill ir_loss from circle_loss
  ir::train::LossInfo ir_loss;
  const circle::LossFn circle_loss = circle_model->lossfn();

  switch (circle_loss)
  {
    case circle::LossFn::LossFn_CATEGORICAL_CROSSENTROPY:
      ir_loss.loss_code = ir::train::LossCode::CategoricalCrossentropy;
      break;
    case circle::LossFn::LossFn_MEAN_SQUARED_ERROR:
      ir_loss.loss_code = ir::train::LossCode::MeanSquaredError;
      break;
    case circle::LossFn::LossFn_SPARSE_CATEGORICAL_CROSSENTROPY:
      // TODO enable this conversion after core support sparse_categorial_crossentropy
      throw std::runtime_error{"'sparse_categorical_crossentropy' is not supported yet"};
    default:
      throw std::runtime_error{"unknown loss function"};
  }

  // TODO update circle schema to support loss reduction type
  return ir_loss;
}
} // namespace

std::unique_ptr<ir::train::TrainingInfo> loadTrainingInfo(const uint8_t *buffer, const size_t size)
{
  flatbuffers::Verifier v(buffer, size);
  bool verified = circle::VerifyModelTrainingBuffer(v);
  if (not verified)
    throw std::runtime_error{"TrainingInfo buffer is not accessible"};

  const circle::ModelTraining *circle_model =
    circle::GetModelTraining(static_cast<const void *>(buffer));

  auto tinfo = std::make_unique<ir::train::TrainingInfo>();
  {
    tinfo->setBatchSize(circle_model->batch_size());
    tinfo->setOptimizerInfo(loadOptimizerInfo(circle_model));
    tinfo->setLossInfo(loadLossInfo(circle_model));
  }
  return tinfo;
}

} // namespace traininfo_loader
} // namespace train
} // namespace onert
